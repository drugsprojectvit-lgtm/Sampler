import hashlib
import math
import os
import re
from typing import Any, Dict, List, Optional, Set

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        PointStruct,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
except ImportError:  # pragma: no cover
    QdrantClient = None


class HybridVectorStoreService:
    """Qdrant-backed hybrid (dense + sparse) vector store."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        dense_dim: int = 384,
    ):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client package is required. Install with `pip install qdrant-client`.")

        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "symbol_records")
        self.dense_dim = dense_dim
        self.client = QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {col.name for col in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=self.dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    def _dense_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dense_dim
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()):
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dense_dim
            sign = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _sparse_embed(self, text: str) -> SparseVector:
        counts: Dict[int, float] = {}
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 100_000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        indices = sorted(counts.keys())
        values = [counts[i] for i in indices]
        return SparseVector(indices=indices, values=values)

    async def upsert_records(self, graph: Dict[str, Dict], changed_ids: Optional[Set[str]] = None):
        changed_ids = changed_ids or set(graph.keys())
        points: List[PointStruct] = []
        for node_id in changed_ids:
            node = graph.get(node_id)
            if not node:
                continue
            symbol = node.get("symbol", {})
            text = f"{symbol.get('name', '')} {symbol.get('docstring', '')} {' '.join(symbol.get('arg_names', []))}"[:4000]
            dense = self._dense_embed(text)
            sparse = self._sparse_embed(text)
            points.append(
                PointStruct(
                    id=node_id,
                    vector={"dense": dense, "sparse": sparse},
                    payload={
                        "text": text,
                        "repo": node.get("repo"),
                        "file": node.get("file"),
                    },
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
