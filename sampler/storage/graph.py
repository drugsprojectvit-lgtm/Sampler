import json
import os
from typing import Any, Dict, List, Optional, Set

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


class GraphStorageService:
    """Neo4j-backed function call graph storage."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is required. Install with `pip install neo4j`.")
        if not self.password:
            raise RuntimeError("NEO4J_PASSWORD must be set to use GraphStorageService.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self._init_schema()

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, params or {}))

    def _init_schema(self):
        self._run("CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE")
        self._run("CREATE INDEX function_repo_file IF NOT EXISTS FOR (f:Function) ON (f.repo, f.file)")
        self._run("CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)")

    def upsert_repo_graph(
        self,
        repo_name: str,
        graph: Dict[str, Dict],
        changed_files: Optional[Set[str]] = None,
    ):
        if changed_files is None:
            changed_files = {node.get("file") for node in graph.values() if node.get("file")}

        if changed_files:
            self._run(
                """
                MATCH (f:Function {repo: $repo})
                WHERE f.file IN $changed_files
                DETACH DELETE f
                """,
                {"repo": repo_name, "changed_files": list(changed_files)},
            )

        for node_id, node in graph.items():
            symbol = node.get("symbol", {})
            params = {
                "id": node_id,
                "repo": repo_name,
                "file": node.get("file"),
                "name": symbol.get("name", node_id),
                "node_type": symbol.get("type", ""),
                "payload": json.dumps(node),
                "calls": symbol.get("calls", []),
            }
            self._run(
                """
                MERGE (f:Function {id: $id})
                SET f.repo = $repo,
                    f.file = $file,
                    f.name = $name,
                    f.node_type = $node_type,
                    f.payload = $payload
                WITH f
                UNWIND $calls AS target_name
                MERGE (t:Function {id: $repo + "::__name__::" + target_name})
                ON CREATE SET t.repo = $repo, t.name = target_name, t.node_type = "placeholder"
                MERGE (f)-[:CALLS]->(t)
                """,
                params,
            )

    def impact_analysis(self, target_name: str, max_depth: int = 4) -> List[str]:
        rows = self._run(
            """
            MATCH (target:Function {name: $target_name})<-[:CALLS*1..$max_depth]-(caller:Function)
            RETURN DISTINCT caller.id AS caller_id
            ORDER BY caller_id
            """,
            {"target_name": target_name, "max_depth": max_depth},
        )
        return [row["caller_id"] for row in rows if row.get("caller_id")]

    def close(self):
        self.driver.close()

    def __del__(self):
        if hasattr(self, "driver"):
            self.driver.close()
