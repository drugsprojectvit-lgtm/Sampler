import os
import json
import asyncio
import shutil
import stat
import errno
import sys
import re
import ast
import hashlib
import time
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict
from git import Repo
from openai import AsyncOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import tiktoken

# Tree-sitter imports with better error handling
HAS_TREE_SITTER = False
try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
    
    # Try to import language bindings
    TREE_SITTER_LANGS = {}
    
    try:
        import tree_sitter_python as tspython
        TREE_SITTER_LANGS['python'] = tspython
    except ImportError:
        pass
    
    try:
        import tree_sitter_javascript as tsjs
        TREE_SITTER_LANGS['javascript'] = tsjs
    except ImportError:
        pass
    
    try:
        import tree_sitter_typescript as tsts
        TREE_SITTER_LANGS['typescript'] = tsts
    except ImportError:
        pass
    
    try:
        import tree_sitter_java as tsjava
        TREE_SITTER_LANGS['java'] = tsjava
    except ImportError:
        pass
    
    try:
        import tree_sitter_go as tsgo
        TREE_SITTER_LANGS['go'] = tsgo
    except ImportError:
        pass
    
    try:
        import tree_sitter_rust as tsrust
        TREE_SITTER_LANGS['rust'] = tsrust
    except ImportError:
        pass
    
    try:
        import tree_sitter_cpp as tscpp
        TREE_SITTER_LANGS['cpp'] = tscpp
    except ImportError:
        pass
    
    try:
        import tree_sitter_c as tsc
        TREE_SITTER_LANGS['c'] = tsc
    except ImportError:
        pass
    
    try:
        import tree_sitter_bash as tsbash
        TREE_SITTER_LANGS['bash'] = tsbash
    except ImportError:
        pass
        
except ImportError:
    pass

# Vector DB imports
try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Jupyter notebook support
try:
    import nbformat
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Graph Configuration
MAX_RECURSION_DEPTH = 3
MAX_CONCURRENCY = 50 
MAX_CONTEXT_CHARS = 200000 
MAX_CONTEXT_TOKENS = 30000
EXTRACT_GLOBALS_LINES = 50  # Number of lines from top of file to extract
INCLUDE_FILE_HEADER = True  # Include file-level docstrings/comments

# Embedding Configuration
EMBEDDING_BATCH_SIZE = 100
TOP_K_SEEDS = 5
BACKWARD_TRAVERSAL_DEPTH = 3
INCLUDE_HIGH_LEVEL_CONTEXT = True
MAX_SUMMARY_TOKENS = 4000
MAX_REASONING_TURNS = 4

# Retrieval weighting configuration
COMMENT_MATCH_WEIGHT = 0.2
COMMENT_DENSITY_WEIGHT = 0.1
INTENT_SIMILARITY_WEIGHT = 0.6
CODE_SIMILARITY_WEIGHT = 0.3
GRAPH_PROXIMITY_WEIGHT = 0.1
CENTRALITY_WEIGHT = 0.2
BM25_WEIGHT = 0.3
EMBEDDING_WEIGHT = 0.5
CONTEXT_COMPRESSION_TRIGGER_TOKENS = 22000
LAST_SELECTOR_NODES: List[str] = []

# Cache Configuration
TENANT_ID = os.getenv("GITHUB_AGENT_TENANT", "default_tenant")
SESSION_ID = os.getenv("GITHUB_AGENT_SESSION", "default_session")
RUNTIME_ROOT = os.path.join("./runtime_data", TENANT_ID, SESSION_ID)

CACHE_DIR = os.path.join(RUNTIME_ROOT, "pipeline_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
GRAPH_FILE = os.path.join(CACHE_DIR, "symbol_graph.json")
MEMORY_FILE = os.path.join(CACHE_DIR, "reasoning_memory.json")
VISUALIZATION_DIR = os.path.join(RUNTIME_ROOT, "visualizations")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found in .env. Please set it.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

# Paths
TEMP_DIR = os.path.join(RUNTIME_ROOT, "temp_session_data")

# --- Helper: Force Delete Read-Only Files ---
def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

# --- Helper: Robust Retry Wrapper ---
async def safe_chat_completion(model, messages, response_format=None, retries=3):
    base_delay = 2
    for attempt in range(retries):
        try:
            kwargs = {"model": model, "messages": messages, "timeout": 45}
            if response_format: kwargs["response_format"] = response_format
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = base_delay * (2 ** attempt)
                print(f"   ⚠️ Rate limit. Pausing {wait}s...", flush=True)
                await asyncio.sleep(wait)
            elif "json" in str(e).lower():
                print(f"   ⚠️ JSON Error. Retrying...", flush=True)
            else:
                if attempt == retries - 1: raise e
    raise Exception("Exceeded max retries.")

# --- Helper: Batch Embeddings ---
async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    if not texts:
        return []
    
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return [[0.0] * 1536 for _ in texts]

# --- Helper: Token Counting ---
def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(tokenizer.encode(text))

def truncate_to_token_budget(texts: List[str], budget: int) -> List[str]:
    """Truncate list of texts to fit within token budget."""
    result = []
    current_tokens = 0
    
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens <= budget:
            result.append(text)
            current_tokens += tokens
        else:
            remaining = budget - current_tokens
            if remaining > 100:
                encoded = tokenizer.encode(text)[:remaining]
                result.append(tokenizer.decode(encoded) + "\n... [truncated]")
            break
    
    return result

# --- Helper: Route Normalization ---
def normalize_route(route: str) -> str:
    """Normalize API routes for matching (handle parameters)."""
    route = re.sub(r'/\d+', '/*', route)
    route = re.sub(r'/:\w+', '/*', route)
    route = re.sub(r'/\{[^}]+\}', '/*', route)
    route = re.sub(r'/\*+', '/*', route)
    return route.lower().strip()


def _extract_preceding_comment_block(lines: List[str], start_line_1_idx: int) -> str:
    """Extract contiguous comments immediately above a symbol start line."""
    idx = max(0, start_line_1_idx - 2)
    comment_lines = []

    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped:
            if comment_lines:
                break
            idx -= 1
            continue

        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or stripped.endswith('*/'):
            comment_lines.append(lines[idx])
            idx -= 1
            continue

        break

    return "\n".join(reversed(comment_lines)).strip()


def _extract_python_docstring(code: str) -> str:
    """Extract first python docstring from a function/class snippet."""
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return ast.get_docstring(tree.body[0]) or ""
    except Exception:
        return ""
    return ""


def _extract_block_docstring(code: str) -> str:
    """Extract JS/Java style block docstring from a code snippet."""
    match = re.search(r"/\*\*(.*?)\*/", code, re.DOTALL)
    return match.group(1).strip() if match else ""


def _comment_keyword_overlap_score(query: str, comment_text: str) -> float:
    """Simple lexical overlap score between query and comment/docs."""
    if not query or not comment_text:
        return 0.0

    q_tokens = set(re.findall(r"[a-zA-Z_]{3,}", query.lower()))
    c_tokens = set(re.findall(r"[a-zA-Z_]{3,}", comment_text.lower()))
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = q_tokens.intersection(c_tokens)
    return len(overlap) / len(q_tokens)




def _tokenize_for_sparse(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower())


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize arbitrary scores to [0, 1] range."""
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    if max_val == min_val:
        return {k: 1.0 for k in scores.keys()}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


TAINT_SOURCES = {"input", "request.get", "req.body", "request.args.get", "request.form.get", "request.json", "sys.argv"}
TAINT_SINKS = {"db.execute", "cursor.execute", "eval", "exec", "subprocess.run", "os.system"}
SANITIZERS = {"sanitize", "escape", "quote", "clean", "validate"}


def _safe_ast_parse(code: str):
    try:
        return ast.parse(code or "")
    except Exception:
        return None


def _name_from_call(call: ast.Call) -> str:
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        base = ""
        if isinstance(fn.value, ast.Name):
            base = fn.value.id + "."
        return f"{base}{fn.attr}"
    return ""


def _extract_type_awareness(code: str) -> Dict[str, Any]:
    tree = _safe_ast_parse(code)
    result = {
        "type_registry": {},
        "interface_links": [],
        "generic_usages": [],
        "field_links": {}
    }
    if tree is None:
        return result

    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            fields = []
            methods = []
            for b in n.body:
                if isinstance(b, ast.FunctionDef):
                    methods.append(b.name)
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    fields.append(b.target.id)
                if isinstance(b, ast.Assign):
                    for t in b.targets:
                        if isinstance(t, ast.Name):
                            fields.append(t.id)
            bases = [getattr(base, 'id', getattr(base, 'attr', '')) for base in n.bases]
            result["type_registry"][n.name] = {"fields": sorted(set(fields)), "methods": sorted(set(methods)), "bases": bases}
            for b in bases:
                if b:
                    result["interface_links"].append({"class": n.name, "implements": b})
        if isinstance(n, ast.Subscript):
            outer = getattr(n.value, 'id', getattr(n.value, 'attr', ''))
            inner = ast.unparse(n.slice) if hasattr(ast, 'unparse') else ""
            if outer and inner:
                result["generic_usages"].append({"container": outer, "inner": inner})
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            key = f"{n.value.id}.{n.attr}"
            result["field_links"].setdefault(key, 0)
            result["field_links"][key] += 1

    return result


def _extract_python_cfg(code: str) -> Dict[str, Any]:
    tree = _safe_ast_parse(code)
    if tree is None:
        return {"blocks": [], "edges": []}

    blocks, edges = [], []
    block_id = 0

    def add_block(kind: str, lineno: int) -> int:
        nonlocal block_id
        bid = block_id
        block_id += 1
        blocks.append({"id": bid, "kind": kind, "lineno": lineno})
        return bid

    prev = None
    for stmt in tree.body:
        kind = type(stmt).__name__
        curr = add_block(kind, getattr(stmt, 'lineno', 0))
        if prev is not None:
            edges.append((prev, curr))
        prev = curr

        if isinstance(stmt, ast.If):
            cond_id = curr
            if stmt.body:
                t_id = add_block("IfBody", getattr(stmt.body[0], 'lineno', 0))
                edges.append((cond_id, t_id))
            if stmt.orelse:
                e_id = add_block("ElseBody", getattr(stmt.orelse[0], 'lineno', 0))
                edges.append((cond_id, e_id))
        if isinstance(stmt, (ast.For, ast.While)):
            loop_entry = curr
            if stmt.body:
                body_id = add_block("LoopBody", getattr(stmt.body[0], 'lineno', 0))
                edges.append((loop_entry, body_id))
                edges.append((body_id, loop_entry))
        if isinstance(stmt, (ast.Return, ast.Raise)):
            end_id = add_block("Exit", getattr(stmt, 'lineno', 0))
            edges.append((curr, end_id))

    return {"blocks": blocks, "edges": edges}


def _extract_python_data_flow(code: str) -> Dict[str, Any]:
    tree = _safe_ast_parse(code)
    data = {
        "defs": {},
        "uses": defaultdict(list),
        "edges": [],
        "return_flows": [],
        "field_dependencies": defaultdict(list),
        "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}
    }
    if tree is None:
        return {
            "defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {},
            "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}
        }

    tainted = set()

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            rhs_names = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
            rhs_attrs = {f"{x.value.id}.{x.attr}" for x in ast.walk(n.value) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)}
            call_name = _name_from_call(n.value) if isinstance(n.value, ast.Call) else ""

            for t in n.targets:
                if isinstance(t, ast.Name):
                    var = t.id
                    data["defs"][var] = getattr(n, 'lineno', 0)
                    for src in rhs_names:
                        data["edges"].append((src, var))
                    for src in rhs_attrs:
                        data["edges"].append((src, var))
                    if any(src in tainted for src in rhs_names) or call_name in TAINT_SOURCES:
                        tainted.add(var)
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    field = f"{t.value.id}.{t.attr}"
                    data["defs"][field] = getattr(n, 'lineno', 0)
                    for src in rhs_names:
                        data["field_dependencies"][field].append(src)

            if isinstance(n.value, ast.Call):
                if call_name in TAINT_SOURCES:
                    data["taint"]["sources"].append({"line": getattr(n, 'lineno', 0), "call": call_name})
                if any(key in call_name for key in SANITIZERS):
                    data["taint"]["sanitizers"].append({"line": getattr(n, 'lineno', 0), "call": call_name})

        if isinstance(n, ast.Call):
            cname = _name_from_call(n)
            arg_names = {x.id for a in n.args for x in ast.walk(a) if isinstance(x, ast.Name)}
            if cname in TAINT_SINKS:
                data["taint"]["sinks"].append({"line": getattr(n, 'lineno', 0), "call": cname})
                if any(a in tainted for a in arg_names):
                    data["taint"]["vulnerabilities"].append({
                        "line": getattr(n, 'lineno', 0),
                        "sink": cname,
                        "tainted_args": sorted([a for a in arg_names if a in tainted])
                    })

        if isinstance(n, ast.Return):
            names = [x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)] if n.value else []
            data["return_flows"].append({"line": getattr(n, 'lineno', 0), "vars": names})

        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            data["uses"][n.id].append(getattr(n, 'lineno', 0))

        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            key = f"{n.value.id}.{n.attr}"
            data["uses"][key].append(getattr(n, 'lineno', 0))

    return {
        "defs": dict(data["defs"]),
        "uses": {k: v for k, v in data["uses"].items()},
        "edges": data["edges"],
        "return_flows": data["return_flows"],
        "field_dependencies": {k: v for k, v in data["field_dependencies"].items()},
        "taint": data["taint"]
    }


def _symbolic_execution_preview(code: str) -> Dict[str, Any]:
    """Lightweight symbolic execution preview with simple path constraints."""
    tree = _safe_ast_parse(code)
    if tree is None:
        return {"constraints": [], "symbols": []}
    constraints, symbols = [], set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            cond = ast.unparse(n.test) if hasattr(ast, 'unparse') else 'condition'
            constraints.append(cond)
            for name in [x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)]:
                symbols.add(name)
    return {"constraints": constraints[:40], "symbols": sorted(symbols)}


def _generate_boundary_test_inputs(arg_names: List[str], arg_types: List[str]) -> List[Dict[str, Any]]:
    """Generate basic boundary test payloads for runtime simulation."""
    if not arg_names:
        return []
    samples = []
    base = {}
    for i, arg in enumerate(arg_names):
        typ = (arg_types[i] if i < len(arg_types) else '').lower()
        if 'int' in typ:
            base[arg] = 0
        elif 'float' in typ:
            base[arg] = 0.0
        elif 'bool' in typ:
            base[arg] = False
        else:
            base[arg] = ""
    samples.append(dict(base))
    alt = {}
    for i, arg in enumerate(arg_names):
        typ = (arg_types[i] if i < len(arg_types) else '').lower()
        if 'int' in typ:
            alt[arg] = 1
        elif 'float' in typ:
            alt[arg] = 1.5
        elif 'bool' in typ:
            alt[arg] = True
        else:
            alt[arg] = "boundary"
    samples.append(alt)
    return samples


def _extract_param_bindings(code: str) -> Dict[str, str]:
    tree = _safe_ast_parse(code)
    if tree is None:
        return {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            return {a.arg: a.arg for a in n.args.args}
    return {}


def _infer_security_role(node_data: Dict[str, Any]) -> str:
    code = (node_data.get('code') or '').lower()
    decorators = [d.lower() for d in node_data.get('decorators', [])]
    if any(tok in code for tok in ["request.", "input(", "sys.argv"]):
        return "source"
    if any(tok in code for tok in ["execute(", "eval(", "exec(", "os.system("]):
        return "sink"
    if any(tok in code for tok in ["sanitize", "escape", "validate"]):
        return "sanitizer"
    if any("auth" in d or "login_required" in d or "requires_auth" in d for d in decorators):
        return "auth_boundary"
    return "neutral"


def _build_package_dependency_metadata(files_data: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    out = {"package_json": {}, "requirements": {}, "all": {}}
    if 'package.json' in files_data:
        try:
            pkg = json.loads(files_data['package.json']['content'])
            out['package_json'] = {
                **pkg.get('dependencies', {}),
                **pkg.get('devDependencies', {})
            }
        except Exception:
            pass
    if 'requirements.txt' in files_data:
        reqs = {}
        for line in files_data['requirements.txt']['content'].splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '==' in line:
                k, v = line.split('==', 1)
                reqs[k.strip()] = v.strip()
            else:
                reqs[line] = '*'
        out['requirements'] = reqs

    merged = {}
    merged.update(out['package_json'])
    merged.update(out['requirements'])
    out['all'] = merged
    return out


def _track_symbol_refactors(active_graph: Dict[str, Dict], previous_graph: Optional[Dict[str, Dict]]):
    if not previous_graph:
        return
    prev_by_hash = defaultdict(list)
    for old_id, old_node in previous_graph.items():
        prev_by_hash[old_node.get('node_hash', '')].append((old_id, old_node))

    for nid, node in active_graph.items():
        nh = node.get('node_hash', '')
        if not nh or nid in previous_graph:
            continue
        cands = prev_by_hash.get(nh, [])
        if cands:
            node['previous_identity'] = cands[0][0]


def _load_reasoning_memory() -> List[Dict[str, Any]]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_reasoning_memory(entries: List[Dict[str, Any]]):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(entries[-200:], f, indent=2)


def _find_memory_match(query: str, threshold: float = 0.84) -> Optional[Dict[str, Any]]:
    best, best_score = None, 0.0
    for item in _load_reasoning_memory():
        q = item.get('query', '')
        score = SequenceMatcher(None, query.lower(), q.lower()).ratio()
        if score > best_score:
            best, best_score = item, score
    if best and best_score >= threshold:
        return best
    return None


def _record_reasoning_memory(query: str, selected_nodes: List[str], answer: str):
    entries = _load_reasoning_memory()
    entries.append({"query": query, "selected_nodes": selected_nodes[:80], "answer": answer[:3000], "ts": time.time()})
    _save_reasoning_memory(entries)


def export_graph_visualizations(multi_graph: Dict[str, Dict], output_dir: str = VISUALIZATION_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for repo, graph in multi_graph.items():
        symbol_dot = ["digraph SymbolGraph {"]
        for nid, node in graph.items():
            label = node.get('symbol', {}).get('name', nid.split('::')[-1]).replace('"', "'")
            symbol_dot.append(f'  "{nid}" [label="{label}"];')
            for dep in node.get('dependencies', []):
                if dep in graph:
                    symbol_dot.append(f'  "{nid}" -> "{dep}";')
        symbol_dot.append("}")
        with open(os.path.join(output_dir, f"{repo}_symbol.dot"), 'w') as f:
            f.write("\n".join(symbol_dot))

        for nid, node in list(graph.items())[:40]:
            dfg = node.get('dfg', {})
            cfg = node.get('cfg', {})
            if dfg.get('edges'):
                dfg_dot = ["digraph DFG {"]
                for src, dst in dfg.get('edges', [])[:200]:
                    dfg_dot.append(f'  "{src}" -> "{dst}";')
                dfg_dot.append("}")
                with open(os.path.join(output_dir, f"{repo}_{hashlib.md5(nid.encode()).hexdigest()[:10]}_dfg.dot"), 'w') as f:
                    f.write("\n".join(dfg_dot))
            if cfg.get('edges'):
                cfg_dot = ["digraph CFG {"]
                for b in cfg.get('blocks', []):
                    cfg_dot.append(f'  "{b.get("id")}" [label="{b.get("kind")}@{b.get("lineno")}"];')
                for src, dst in cfg.get('edges', [])[:200]:
                    cfg_dot.append(f'  "{src}" -> "{dst}";')
                cfg_dot.append("}")
                with open(os.path.join(output_dir, f"{repo}_{hashlib.md5(nid.encode()).hexdigest()[:10]}_cfg.dot"), 'w') as f:
                    f.write("\n".join(cfg_dot))

def _extract_external_lib_from_import(import_stmt: str) -> Optional[str]:
    """Extract external library/module root from common import syntaxes."""
    stmt = (import_stmt or "").strip()
    py_from = re.search(r"from\s+([\w\.]+)\s+import", stmt)
    if py_from:
        return py_from.group(1).split('.')[0]
    py_imp = re.search(r"import\s+([\w\.]+)", stmt)
    if py_imp:
        return py_imp.group(1).split('.')[0]
    js_from = re.search(r"from\s+[\"']([^\"']+)[\"']", stmt)
    if js_from:
        mod = js_from.group(1)
        if not mod.startswith('.'):
            return mod.split('/')[0]
    req = re.search(r"require\s*\(\s*[\"']([^\"']+)[\"']\s*\)", stmt)
    if req:
        mod = req.group(1)
        if not mod.startswith('.'):
            return mod.split('/')[0]
    return None


def _comment_contextual_boost(query: str, comment_text: str) -> float:
    """Contextual boost for domain-specific explanatory comments."""
    if not query or not comment_text:
        return 0.0
    q = query.lower()
    c = comment_text.lower()
    domain_terms = ["auth", "payment", "cache", "db", "database", "queue", "token", "session", "validator"]
    boost = 0.0
    for t in domain_terms:
        if t in q and t in c:
            boost += 0.05
    if "example" in c or "e.g." in c or "for example" in c:
        boost += 0.05
    if "generated" in c or "auto-generated" in c:
        boost -= 0.05
    return max(-0.1, min(boost, 0.2))


def _compute_node_content_hash(node: Dict[str, Any]) -> str:
    base = "|".join([
        str(node.get('type', '')),
        str(node.get('file', '')),
        str(node.get('code', '')),
        str(node.get('docstring', '')),
        str(node.get('comments_above', '')),
        str(node.get('arg_types', [])),
        str(node.get('return_type', '')),
    ])
    return hashlib.md5(base.encode('utf-8', errors='ignore')).hexdigest()

def _comment_quality_score(docstring: str, comments: str) -> float:
    """Estimate documentation quality beyond raw density."""
    text = f"{docstring}\n{comments}".strip()
    if not text:
        return 0.0

    score = 0.2
    lower = text.lower()

    # structured documentation boosts
    if "args:" in lower or "returns:" in lower or "raises:" in lower:
        score += 0.2
    if "@param" in lower or "@returns" in lower or "@throws" in lower:
        score += 0.2

    # penalize low-signal and generated patterns
    penalties = ["todo", "fixme", "autogenerated", "auto-generated", "generated by", "boilerplate"]
    if any(p in lower for p in penalties):
        score -= 0.2

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        unique_ratio = len(set(lines)) / max(len(lines), 1)
        if unique_ratio < 0.6:
            score -= 0.15

    # very short comments are often low intent signal
    if len(re.findall(r"[a-zA-Z_]{3,}", text)) < 8:
        score -= 0.05

    return max(0.0, min(score, 1.0))


def _global_node_id(repo: str, file_path: str, symbol_name: str) -> str:
    """Stable globally unique node id across repositories."""
    return f"gid::{repo}::{file_path}::{symbol_name}"


def _extract_python_semantic_metadata(code: str) -> Dict[str, Any]:
    """Extract lightweight data-flow related metadata from Python functions."""
    meta = {
        "arg_names": [],
        "arg_types": [],
        "return_type": None,
        "variable_mentions": [],
        "exception_types": [],
        "decorators": []
    }
    try:
        tree = ast.parse(code)
        if not tree.body:
            return meta
        node = tree.body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            meta["arg_names"] = [a.arg for a in node.args.args]
            meta["arg_types"] = [ast.unparse(a.annotation) for a in node.args.args if getattr(a, 'annotation', None)]
            if node.returns:
                meta["return_type"] = ast.unparse(node.returns)
            meta["decorators"] = [ast.unparse(d) for d in node.decorator_list]
            var_names = set()
            exc_types = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    var_names.add(sub.id)
                elif isinstance(sub, ast.ExceptHandler) and sub.type is not None:
                    try:
                        exc_types.add(ast.unparse(sub.type))
                    except Exception:
                        pass
            meta["variable_mentions"] = sorted(var_names)[:50]
            meta["exception_types"] = sorted(exc_types)
    except Exception:
        pass
    return meta


def reconstruct_file_slice(file_globals: str, nodes: List[Dict[str, Any]], anchor_ids: Set[str] = None) -> str:
    """Builds a smart slice: Full code for anchors, signatures for context."""
    block = f"--- FILE GLOBALS ---\n{file_globals}\n" if file_globals else ""
    anchor_ids = anchor_ids or set()

    for node in nodes:
        name = node.get('symbol', {}).get('name', 'unknown')
        is_anchor = node.get('global_id') in anchor_ids or node.get('node_id') in anchor_ids

        if is_anchor:
            block += f"\n// FULL SYMBOL: {name}\n{node.get('code', '')}\n"
        else:
            # SKELETAL VIEW: Just the signature and docstring
            doc = node.get('docstring', '').split('\n')[0]
            block += f"\n// CONTEXT ONLY: {node.get('type')} {name}\n// Doc: {doc}\n// [Implementation hidden to save tokens]\n"
    return block


def compute_file_content_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()


def compute_multi_repo_file_hashes(multi_repo_data: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    """Compute file-level hashes for partial re-indexing."""
    result = {}
    for repo_name, files_data in multi_repo_data.items():
        result[repo_name] = {
            filename: compute_file_content_hash(data.get('content', ''))
            for filename, data in files_data.items()
        }
    return result

def init_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"📁 Cache directory: {CACHE_DIR}")

# --- Cleanup ---
def perform_cleanup():
    print("\n🧹 Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
        except: pass
    print("✅ Cleanup complete.")

# --- Repository Hashing for Cache Invalidation ---
def compute_repo_hash(repo_path: str) -> str:
    """Compute hash of all tracked files in repo for cache validation."""
    hasher = hashlib.md5()
    try:
        repo = Repo(repo_path)
        for item in repo.tree().traverse():
            if item.type == 'blob':
                hasher.update(item.path.encode())
                hasher.update(str(item.hexsha).encode())
    except:
        for root, dirs, files in os.walk(repo_path):
            for f in sorted(files):
                hasher.update(f.encode())
    return hasher.hexdigest()

# --- 1. Universal File Loader ---

def is_valid_file(filename):
    ALWAYS_KEEP_NAMES = {
        'dockerfile', 'makefile', 'gemfile', 'jenkinsfile', 'procfile', 
        'requirements.txt', 'package.json', 'cargo.toml', 'go.mod', 'pom.xml',
        'tsconfig.json', 'go.sum', 'package-lock.json'
    }
    ALLOWED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.cs', 
        '.go', '.rs', '.rb', '.php', '.c', '.cc', '.hpp', '.sh', '.bash',
        '.ipynb', '.ino'  # Added Jupyter notebooks and Arduino files
    }
    name = os.path.basename(filename).lower()
    ext = os.path.splitext(filename)[1].lower()
    return name in ALWAYS_KEEP_NAMES or ext in ALLOWED_EXTENSIONS

def read_universal_text(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f: 
            return f.read()
    except: 
        return ""

def extract_notebook_code(notebook_path: str) -> str:
    """Extract code cells from Jupyter notebook."""
    if not HAS_NBFORMAT:
        return read_universal_text(notebook_path)
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                code_cells.append(f"# Cell {i+1}\n{cell.source}")
        
        return "\n\n".join(code_cells)
    except Exception as e:
        print(f"⚠️ Error reading notebook {notebook_path}: {e}")
        return read_universal_text(notebook_path)

async def async_read_file(path, relative_path):
    """Read file with special handling for notebooks."""
    if path.endswith('.ipynb'):
        return await asyncio.to_thread(extract_notebook_code, path)
    return await asyncio.to_thread(read_universal_text, path)

async def handle_github_repo(url, source_id):
    if not url: return None, None
    
    clean_name = url.split("/")[-1].replace(".git", "")
    if not clean_name: clean_name = f"repo_{source_id}"
    
    repo_path = os.path.join(TEMP_DIR, f"{clean_name}_{source_id}")
    
    print(f"🔄 Cloning {clean_name}...")
    try:
        await asyncio.to_thread(Repo.clone_from, url, repo_path)
        return repo_path, clean_name
    except Exception as e:
        print(f"❌ Git Clone Failed for {url}: {e}")
        return None, None

async def ingest_sources(github_inputs: str):
    if os.path.exists(TEMP_DIR): 
        shutil.rmtree(TEMP_DIR, onerror=handle_remove_readonly)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    git_urls = [s.strip() for s in github_inputs.split(',') if s.strip()]
    
    dir_tasks = []
    for i, url in enumerate(git_urls):
        dir_tasks.append(handle_github_repo(url, i))
        
    repo_results = await asyncio.gather(*dir_tasks)
    
    multi_repo_data = {}
    read_tasks = []
    repo_hashes = {}
    
    for repo_path, repo_name in repo_results:
        if not repo_path: continue
        
        if repo_name not in multi_repo_data:
            multi_repo_data[repo_name] = {}
            repo_hashes[repo_name] = compute_repo_hash(repo_path)
            
        for root, _, files in os.walk(repo_path):
            if ".git" in root: continue
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                
                if is_valid_file(file):
                    read_tasks.append((async_read_file(full_path, rel_path), repo_name, rel_path))

    print(f"\n📖 Reading files across {len(multi_repo_data)} repositories...")
    
    file_contents = await asyncio.gather(*[t[0] for t in read_tasks])
    
    total_files = 0
    for i, content in enumerate(file_contents):
        _, r_name, r_path = read_tasks[i]
        if content and content.strip():
            multi_repo_data[r_name][r_path] = {"content": content}
            total_files += 1

    print(f"✅ Total Loaded: {total_files} files across {list(multi_repo_data.keys())}.")
    return multi_repo_data, repo_hashes

class GlobalExtractor:
    """
    Extracts global variables, constants, configurations, and defines from files.
    CRITICAL for understanding code context beyond just function definitions.
    """

    def extract_globals_from_content(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Extract global-level declarations.
        Returns: {header, imports, constants, globals, config, defines}
        """
        ext = os.path.splitext(filename)[1].lower()
        
        result = {
            "header": "",
            "imports": [],
            "constants": [],
            "globals": [],
            "config": [],
            "defines": []
        }
        
        lines = content.split('\n')
        
        if INCLUDE_FILE_HEADER:
            result["header"] = self._extract_header(lines, ext)
        
        if ext == '.py':
            result.update(self._extract_python_globals(content, lines))
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            result.update(self._extract_js_globals(content, lines))
        elif ext in ['.cpp', '.cc', '.h', '.hpp', '.c', '.ino']:
            result.update(self._extract_cpp_globals(content, lines))
        elif ext in ['.java']:
            result.update(self._extract_java_globals(content, lines))
        elif ext in ['.go']:
            result.update(self._extract_go_globals(content, lines))
        else:
            result.update(self._extract_generic_globals(content, lines))
        
        return result

    def _extract_header(self, lines: List[str], ext: str) -> str:
        """Extract file-level documentation."""
        header_lines = []
        
        if ext == '.py':
            if len(lines) > 0 and lines[0].strip().startswith('"""'):
                for line in lines[:20]:
                    header_lines.append(line)
                    if line.strip().endswith('"""') and len(header_lines) > 1:
                        break
        
        elif ext in ['.cpp', '.c', '.h', '.hpp', '.ino', '.js', '.jsx', '.ts', '.tsx']:
            for line in lines[:30]:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                    header_lines.append(line)
                elif header_lines and not stripped:
                    continue
                else:
                    break
        
        return '\n'.join(header_lines)

    def _extract_python_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Python globals."""
        imports, constants, globals_vars, config = [], [], [], []
        
        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.get_source_segment(content, node))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            var_line = ast.get_source_segment(content, node)
                            if var_name.isupper():
                                constants.append(var_line)
                            elif 'CONFIG' in var_name or 'SETTINGS' in var_name:
                                config.append(var_line)
                            else:
                                globals_vars.append(var_line)
        except:
            for line in lines[:EXTRACT_GLOBALS_LINES]:
                if re.match(r'^import\s+|^from\s+', line.strip()):
                    imports.append(line)
                elif re.match(r'^[A-Z_]+\s*=', line.strip()):
                    constants.append(line)
                elif re.match(r'^[a-z_]\w*\s*=', line.strip()):
                    globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "config": config}

    def _extract_js_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract JavaScript/TypeScript globals."""
        imports, constants, globals_vars, config = [], [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('require('):
                imports.append(line)
            elif re.match(r'const\s+[A-Z_]+', stripped):
                constants.append(line)
            elif 'config' in stripped.lower() or 'settings' in stripped.lower():
                config.append(line)
            elif re.match(r'(let|var|const)\s+\w+', stripped):
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "config": config}

    def _extract_cpp_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract C/C++ globals and defines."""
        imports, constants, globals_vars, defines = [], [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('#include'):
                imports.append(line)
            elif stripped.startswith('#define'):
                defines.append(line)
            elif 'const ' in stripped:
                constants.append(line)
            elif re.match(r'(extern|static)?\s*(int|float|double|char|bool|void|uint\w*|String)', stripped):
                if not stripped.endswith(';'):
                    idx = lines.index(line)
                    multi = line
                    for i in range(1, 5):
                        if idx + i < len(lines):
                            multi += '\n' + lines[idx + i]
                            if ';' in lines[idx + i]:
                                break
                    globals_vars.append(multi)
                else:
                    globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars, "defines": defines}

    def _extract_java_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Java globals."""
        imports, constants, globals_vars = [], [], []
        in_class = False
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(line)
            elif 'class ' in stripped:
                in_class = True
            elif in_class and 'static final' in stripped:
                constants.append(line)
            elif in_class and 'static ' in stripped:
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars}

    def _extract_go_globals(self, content: str, lines: List[str]) -> Dict:
        """Extract Go globals."""
        imports, constants, globals_vars = [], [], []
        
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(line)
            elif stripped.startswith('const '):
                constants.append(line)
            elif stripped.startswith('var '):
                globals_vars.append(line)
        
        return {"imports": imports, "constants": constants, "globals": globals_vars}

    def _extract_generic_globals(self, content: str, lines: List[str]) -> Dict:
        """Generic extraction."""
        globals_vars = []
        for line in lines[:EXTRACT_GLOBALS_LINES]:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                globals_vars.append(line)
        return {"globals": globals_vars}

    def format_globals_for_context(self, globals_dict: Dict[str, List[str]]) -> str:
        """Format extracted globals into readable context."""
        sections = []
        
        if globals_dict.get("header"):
            sections.append(f"# FILE HEADER\n{globals_dict['header']}\n")
        if globals_dict.get("imports"):
            sections.append(f"# IMPORTS\n" + "\n".join(globals_dict['imports']) + "\n")
        if globals_dict.get("defines"):
            sections.append(f"# DEFINES\n" + "\n".join(globals_dict['defines']) + "\n")
        if globals_dict.get("constants"):
            sections.append(f"# CONSTANTS\n" + "\n".join(globals_dict['constants']) + "\n")
        if globals_dict.get("config"):
            sections.append(f"# CONFIGURATION\n" + "\n".join(globals_dict['config']) + "\n")
        if globals_dict.get("globals"):
            sections.append(f"# GLOBAL VARIABLES\n" + "\n".join(globals_dict['globals']) + "\n")
        
        return "\n".join(sections) if sections else ""

# --- 2. Enhanced Tree-Sitter Parser ---

class TreeSitterParser:
    """
    Production-grade parser using Tree-sitter with fallback to regex.
    """
    
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self.global_extractor = GlobalExtractor()
        
        if not HAS_TREE_SITTER:
            print("⚠️ Tree-sitter not available, using regex fallback")
            return
        
        # Initialize available languages
        self._init_languages()
        
        # Regex fallback patterns for unsupported languages
        self.REGEX_PATTERNS = {
            'cpp': [
                r'(?:void|int|float|double|bool|char|auto)\s+(\w+)\s*\([^)]*\)\s*\{',
                r'(\w+)\s*::\s*(\w+)\s*\([^)]*\)\s*\{',  # Class methods
                r'class\s+(\w+)',
            ],
            'c': [
                r'(?:void|int|float|double|char|static|extern)\s+(\w+)\s*\([^)]*\)\s*\{',
            ],
            'sh': [
                r'function\s+(\w+)\s*\(\s*\)\s*\{',
                r'(\w+)\s*\(\s*\)\s*\{',
            ],
            'bash': [
                r'function\s+(\w+)\s*\(\s*\)\s*\{',
                r'(\w+)\s*\(\s*\)\s*\{',
            ],
            'ino': [  # Arduino (C++ based)
                r'(?:void|int|float|double|bool|char)\s+(\w+)\s*\([^)]*\)\s*\{',
            ],
        }
        
        # API patterns
        self.API_CALL_PATTERNS = [
            r'fetch\s*\(\s*["\']([^"\']+)["\']',
            r'axios\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'http\.(?:Get|Post|Put|Delete)\s*\(\s*["\']([^"\']+)["\']',
            r'requests\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        ]
        
        self.API_ROUTE_PATTERNS = [
            r'@app\.(?:get|post|put|delete|route)\s*\(\s*["\']([^"\']+)["\']',
            r'@router\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'@(?:Get|Post|Put|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
            r'app\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
            r'router\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
        ]

    def _init_languages(self):
        """Initialize Tree-sitter language parsers."""
        try:
            # Python
            if 'python' in TREE_SITTER_LANGS:
                PY_LANGUAGE = Language(TREE_SITTER_LANGS['python'].language())
                self.languages['python'] = PY_LANGUAGE
                self.parsers['.py'] = Parser(PY_LANGUAGE)
            
            # JavaScript
            if 'javascript' in TREE_SITTER_LANGS:
                JS_LANGUAGE = Language(TREE_SITTER_LANGS['javascript'].language())
                self.languages['javascript'] = JS_LANGUAGE
                js_parser = Parser(JS_LANGUAGE)
                self.parsers['.js'] = js_parser
                self.parsers['.jsx'] = js_parser
            
            # TypeScript
            if 'typescript' in TREE_SITTER_LANGS:
                TS_LANGUAGE = Language(TREE_SITTER_LANGS['typescript'].language_typescript())
                self.languages['typescript'] = TS_LANGUAGE
                self.parsers['.ts'] = Parser(TS_LANGUAGE)
                
                TSX_LANGUAGE = Language(TREE_SITTER_LANGS['typescript'].language_tsx())
                self.parsers['.tsx'] = Parser(TSX_LANGUAGE)
            
            # Java
            if 'java' in TREE_SITTER_LANGS:
                JAVA_LANGUAGE = Language(TREE_SITTER_LANGS['java'].language())
                self.languages['java'] = JAVA_LANGUAGE
                self.parsers['.java'] = Parser(JAVA_LANGUAGE)
            
            # Go
            if 'go' in TREE_SITTER_LANGS:
                GO_LANGUAGE = Language(TREE_SITTER_LANGS['go'].language())
                self.languages['go'] = GO_LANGUAGE
                self.parsers['.go'] = Parser(GO_LANGUAGE)
            
            # Rust
            if 'rust' in TREE_SITTER_LANGS:
                RUST_LANGUAGE = Language(TREE_SITTER_LANGS['rust'].language())
                self.languages['rust'] = RUST_LANGUAGE
                self.parsers['.rs'] = Parser(RUST_LANGUAGE)
            
            # C++
            if 'cpp' in TREE_SITTER_LANGS:
                CPP_LANGUAGE = Language(TREE_SITTER_LANGS['cpp'].language())
                self.languages['cpp'] = CPP_LANGUAGE
                cpp_parser = Parser(CPP_LANGUAGE)
                self.parsers['.cpp'] = cpp_parser
                self.parsers['.cc'] = cpp_parser
                self.parsers['.hpp'] = cpp_parser
                self.parsers['.ino'] = cpp_parser  # Arduino
            
            # C
            if 'c' in TREE_SITTER_LANGS:
                C_LANGUAGE = Language(TREE_SITTER_LANGS['c'].language())
                self.languages['c'] = C_LANGUAGE
                c_parser = Parser(C_LANGUAGE)
                self.parsers['.c'] = c_parser
                self.parsers['.h'] = c_parser
            
            # Bash
            if 'bash' in TREE_SITTER_LANGS:
                BASH_LANGUAGE = Language(TREE_SITTER_LANGS['bash'].language())
                self.languages['bash'] = BASH_LANGUAGE
                bash_parser = Parser(BASH_LANGUAGE)
                self.parsers['.sh'] = bash_parser
                self.parsers['.bash'] = bash_parser
            
            if self.parsers:
                print(f"✅ Tree-sitter initialized for: {list(set(ext for ext in self.parsers.keys()))}")
            
        except Exception as e:
            print(f"⚠️ Tree-sitter initialization error: {e}")
            print("   Falling back to regex parsing for all languages")

    def parse(self, filename: str, content: str) -> Dict[str, Any]:
        """Parse file and extract symbols WITH enhanced global extraction."""
        ext = os.path.splitext(filename)[1].lower()
        # FIRST: Extract globals using enhanced extractor
        globals_data = self.global_extractor.extract_globals_from_content(content, filename)
        formatted_globals = self.global_extractor.format_globals_for_context(globals_data)

        # THEN: Extract functions using tree-sitter or regex
        parser = self.parsers.get(ext)
        if parser:
            try:
                tree = parser.parse(bytes(content, "utf8"))
                result = self._extract_symbols(tree.root_node, content, ext)
            except Exception as e:
                print(f"⚠️ Tree-sitter parse error in {filename}: {e}, using regex fallback")
                result = self._parse_regex_fallback(content, ext, filename)
        else:
            result = self._parse_regex_fallback(content, ext, filename)

        # COMBINE: Replace simple imports with rich global context
        result['globals'] = formatted_globals  # This now includes MUCH more!
        result['globals_data'] = globals_data  # Store structured data too

        return result

    def _extract_symbols(self, root_node, content: str, ext: str) -> Dict[str, Any]:
        """Extract functions, classes, and imports from AST."""
        if ext == '.py':
            return self._extract_python(root_node, content)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return self._extract_javascript(root_node, content)
        elif ext == '.java':
            return self._extract_java(root_node, content)
        elif ext == '.go':
            return self._extract_go(root_node, content)
        elif ext == '.rs':
            return self._extract_rust(root_node, content)
        elif ext in ['.cpp', '.cc', '.hpp', '.c', '.h', '.ino']:
            return self._extract_cpp(root_node, content)
        elif ext in ['.sh', '.bash']:
            return self._extract_bash(root_node, content)
        else:
            return {"nodes": [], "imports": [], "globals": ""}

    def _get_text(self, node, content: str) -> str:
        """Extract text from a node."""
        return content[node.start_byte:node.end_byte]


    def _extract_symbol_docs(self, node, content: str, code: str, language_hint: str) -> Dict[str, Any]:
        """Extract per-symbol structured comments/docstrings for ranking and embedding."""
        lines = content.split('\n')
        start_line = getattr(node, 'start_point', (0, 0))[0] + 1
        comments_above = _extract_preceding_comment_block(lines, start_line)

        if language_hint == 'python':
            docstring = _extract_python_docstring(code)
        else:
            docstring = _extract_block_docstring(code)

        comment_tokens = len(re.findall(r"[a-zA-Z_]+", f"{docstring}\n{comments_above}"))
        code_tokens = max(len(re.findall(r"[a-zA-Z_]+", code)), 1)
        comment_ratio = comment_tokens / code_tokens
        comment_quality = _comment_quality_score(docstring, comments_above)

        return {
            "docstring": docstring,
            "comments_above": comments_above,
            "comment_ratio": comment_ratio,
            "comment_quality": comment_quality
        }

    def _extract_semantic_metadata_from_node(self, node, content: str) -> Dict[str, Any]:
        """Language-agnostic semantic metadata extractor using tree-sitter fields."""
        meta = {
            "arg_names": [],
            "arg_types": [],
            "return_type": None,
            "variable_mentions": [],
            "exception_types": [],
            "decorators": []
        }

        params = node.child_by_field_name('parameters')
        if params:
            for child in params.children:
                if child.type in ['identifier', 'required_parameter', 'optional_parameter', 'parameter_declaration', 'formal_parameter']:
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        meta['arg_names'].append(self._get_text(name_node, content))
                    type_node = child.child_by_field_name('type')
                    if type_node:
                        meta['arg_types'].append(self._get_text(type_node, content))

        ret = node.child_by_field_name('return_type')
        if ret:
            meta['return_type'] = self._get_text(ret, content)

        vars_seen, exc_seen = set(), set()
        def walk(n):
            if n.type in ['identifier', 'type_identifier']:
                vars_seen.add(self._get_text(n, content))
            if n.type in ['throw_statement', 'except_clause', 'catch_clause']:
                exc_seen.add(n.type)
            for c in n.children:
                walk(c)
        walk(node)
        meta['variable_mentions'] = sorted(vars_seen)[:50]
        meta['exception_types'] = sorted(exc_seen)
        return meta

    def _extract_python_bases(self, class_node, content: str) -> List[str]:
        bases = []
        sup = class_node.child_by_field_name('superclasses')
        if sup:
            for ch in sup.children:
                if ch.type in ['identifier', 'attribute']:
                    bases.append(self._get_text(ch, content).split('.')[-1])
        return list(dict.fromkeys(bases))

    def _extract_class_bases_generic(self, class_node, content: str) -> List[str]:
        bases = []
        for field_name in ['superclass', 'interfaces', 'extends_clause', 'implements_clause']:
            n = class_node.child_by_field_name(field_name)
            if n:
                for ch in n.children:
                    if ch.type in ['identifier', 'type_identifier', 'scoped_identifier']:
                        bases.append(self._get_text(ch, content).split('.')[-1])
        return list(dict.fromkeys(bases))

    def _extract_python(self, root, content: str) -> Dict[str, Any]:
        """Extract Python symbols."""
        nodes = []
        imports = []

        def visit(node, namespace=""):
            if node.type in ['import_statement', 'import_from_statement']:
                imports.append(self._get_text(node, content))

            elif node.type == 'function_definition':
                func_name_node = node.child_by_field_name('name')
                if func_name_node:
                    name = self._get_text(func_name_node, content)
                    full_name = f"{namespace}.{name}" if namespace else name
                    code = self._get_text(node, content)
                    calls = self._extract_calls_python(node, content)
                    api_route = self._extract_python_route(node, content)
                    api_calls = self._extract_api_calls(code)
                    comment_meta = self._extract_symbol_docs(node, content, code, 'python')
                    semantic_meta = _extract_python_semantic_metadata(code)

                    nodes.append({
                        "name": full_name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": api_route,
                        "api_outbound": api_calls,
                        "bases": [],
                        **comment_meta,
                        **semantic_meta
                    })

            elif node.type == 'class_definition':
                class_name_node = node.child_by_field_name('name')
                if class_name_node:
                    class_name = self._get_text(class_name_node, content)
                    full_name = f"{namespace}.{class_name}" if namespace else class_name
                    bases = self._extract_python_bases(node, content)
                    class_code = self._get_text(node, content)
                    comment_meta = self._extract_symbol_docs(node, content, class_code, 'python')
                    nodes.append({
                        "name": full_name,
                        "type": "class",
                        "code": class_code,
                        "calls": [],
                        "api_route": None,
                        "api_outbound": [],
                        "bases": bases,
                        **comment_meta,
                        "arg_names": [],
                        "arg_types": [],
                        "return_type": None,
                        "variable_mentions": [],
                        "exception_types": [],
                        "decorators": []
                    })
                    for child in node.children:
                        visit(child, full_name)
                    return

            for child in node.children:
                visit(child, namespace)

        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_python(self, func_node, content: str) -> List[str]:
        """Extract function calls from Python function."""
        calls = set()
        
        def visit(node):
            if node.type == 'call':
                func_node = node.child_by_field_name('function')
                if func_node:
                    if func_node.type == 'identifier':
                        calls.add(self._get_text(func_node, content))
                    elif func_node.type == 'attribute':
                        attr = func_node.child_by_field_name('attribute')
                        if attr:
                            calls.add(self._get_text(attr, content))
            
            for child in node.children:
                visit(child)
        
        visit(func_node)
        return list(calls)

    def _extract_python_route(self, func_node, content: str) -> Optional[str]:
        """Extract API route from Python decorators."""
        prev = func_node.prev_sibling
        while prev and prev.type == 'decorator':
            decorator_text = self._get_text(prev, content)
            for pattern in self.API_ROUTE_PATTERNS:
                match = re.search(pattern, decorator_text)
                if match:
                    return normalize_route(match.group(1))
            prev = prev.prev_sibling
        return None

    def _extract_javascript(self, root, content: str) -> Dict[str, Any]:
        """Extract JavaScript/TypeScript symbols."""
        nodes = []
        imports = []

        def visit(node, namespace=""):
            if node.type in ['import_statement', 'import_clause']:
                imports.append(self._get_text(node, content))

            elif node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_js(node, content)
                    api_calls = self._extract_api_calls(code)
                    comment_meta = self._extract_symbol_docs(node, content, code, 'javascript')
                    semantic_meta = self._extract_semantic_metadata_from_node(node, content)
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls,
                        "bases": [],
                        **comment_meta,
                        **semantic_meta
                    })

            elif node.type == 'lexical_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator':
                        name_node = child.child_by_field_name('name')
                        value_node = child.child_by_field_name('value')
                        if name_node and value_node and value_node.type in ['arrow_function', 'function']:
                            name = self._get_text(name_node, content)
                            code = self._get_text(child, content)
                            calls = self._extract_calls_js(value_node, content)
                            api_calls = self._extract_api_calls(code)
                            comment_meta = self._extract_symbol_docs(child, content, code, 'javascript')
                            semantic_meta = self._extract_semantic_metadata_from_node(value_node, content)
                            nodes.append({
                                "name": name,
                                "type": "function",
                                "code": code,
                                "calls": calls,
                                "api_route": None,
                                "api_outbound": api_calls,
                                "bases": [],
                                **comment_meta,
                                **semantic_meta
                            })

            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    bases = self._extract_class_bases_generic(node, content)
                    class_code = self._get_text(node, content)
                    comment_meta = self._extract_symbol_docs(node, content, class_code, 'javascript')
                    nodes.append({
                        "name": class_name,
                        "type": "class",
                        "code": class_code,
                        "calls": [],
                        "api_route": None,
                        "api_outbound": [],
                        "bases": bases,
                        **comment_meta,
                        "arg_names": [], "arg_types": [], "return_type": None,
                        "variable_mentions": [], "exception_types": [], "decorators": []
                    })
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            if child.type == 'method_definition':
                                method_name_node = child.child_by_field_name('name')
                                if method_name_node:
                                    method_name = self._get_text(method_name_node, content)
                                    full_name = f"{class_name}.{method_name}"
                                    code = self._get_text(child, content)
                                    calls = self._extract_calls_js(child, content)
                                    api_calls = self._extract_api_calls(code)
                                    comment_meta = self._extract_symbol_docs(child, content, code, 'javascript')
                                    semantic_meta = self._extract_semantic_metadata_from_node(child, content)
                                    nodes.append({
                                        "name": full_name,
                                        "type": "method",
                                        "code": code,
                                        "calls": calls,
                                        "api_route": None,
                                        "api_outbound": api_calls,
                                        "bases": [],
                                        **comment_meta,
                                        **semantic_meta
                                    })

            for child in node.children:
                visit(child, namespace)

        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_js(self, node, content: str) -> List[str]:
        """Extract function calls from JS/TS code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'member_expression':
                        prop = func.child_by_field_name('property')
                        if prop:
                            calls.add(self._get_text(prop, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_java(self, root, content: str) -> Dict[str, Any]:
        """Extract Java symbols."""
        nodes = []
        imports = []

        def visit(node, class_context=""):
            if node.type == 'import_declaration':
                imports.append(self._get_text(node, content))

            elif node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    full_name = f"{class_context}.{name}" if class_context else name
                    code = self._get_text(node, content)
                    calls = self._extract_calls_java(node, content)
                    api_route = self._extract_spring_route(node, content)
                    api_calls = self._extract_api_calls(code)
                    comment_meta = self._extract_symbol_docs(node, content, code, 'other')
                    semantic_meta = self._extract_semantic_metadata_from_node(node, content)
                    nodes.append({
                        "name": full_name,
                        "type": "method",
                        "code": code,
                        "calls": calls,
                        "api_route": api_route,
                        "api_outbound": api_calls,
                        "bases": [],
                        **comment_meta,
                        **semantic_meta
                    })

            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    bases = self._extract_class_bases_generic(node, content)
                    class_code = self._get_text(node, content)
                    comment_meta = self._extract_symbol_docs(node, content, class_code, 'other')
                    nodes.append({
                        "name": class_name,
                        "type": "class",
                        "code": class_code,
                        "calls": [],
                        "api_route": None,
                        "api_outbound": [],
                        "bases": bases,
                        **comment_meta,
                        "arg_names": [], "arg_types": [], "return_type": None,
                        "variable_mentions": [], "exception_types": [], "decorators": []
                    })
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            visit(child, class_name)
                    return

            for child in node.children:
                visit(child, class_context)

        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_java(self, node, content: str) -> List[str]:
        """Extract method calls from Java code."""
        calls = set()
        
        def visit(n):
            if n.type == 'method_invocation':
                name_node = n.child_by_field_name('name')
                if name_node:
                    calls.add(self._get_text(name_node, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_spring_route(self, method_node, content: str) -> Optional[str]:
        """Extract Spring @RequestMapping, @GetMapping, etc."""
        prev = method_node.prev_sibling
        while prev:
            if prev.type == 'marker_annotation' or prev.type == 'annotation':
                annot_text = self._get_text(prev, content)
                patterns = [
                    r'@(?:Get|Post|Put|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
                    r'@RequestMapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']'
                ]
                for pattern in patterns:
                    match = re.search(pattern, annot_text)
                    if match:
                        return normalize_route(match.group(1))
            prev = prev.prev_sibling
        return None

    def _extract_go(self, root, content: str) -> Dict[str, Any]:
        """Extract Go symbols."""
        nodes = []
        imports = []

        def visit(node):
            if node.type == 'import_declaration':
                imports.append(self._get_text(node, content))

            elif node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_go(node, content)
                    api_calls = self._extract_api_calls(code)
                    comment_meta = self._extract_symbol_docs(node, content, code, 'other')
                    semantic_meta = self._extract_semantic_metadata_from_node(node, content)
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls,
                        "bases": [],
                        **comment_meta,
                        **semantic_meta
                    })

            for child in node.children:
                visit(child)

        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_go(self, node, content: str) -> List[str]:
        """Extract function calls from Go code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'selector_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_rust(self, root, content: str) -> Dict[str, Any]:
        """Extract Rust symbols."""
        nodes = []
        imports = []
        
        def visit(node):
            if node.type == 'use_declaration':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_rust(node, content)
                    api_calls = self._extract_api_calls(code)
                    comment_meta = self._extract_symbol_docs(node, content, code, 'other')
                    
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls,
                        **comment_meta
                    })
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_calls_rust(self, node, content: str) -> List[str]:
        """Extract function calls from Rust code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'field_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_cpp(self, root, content: str) -> Dict[str, Any]:
        """Extract C/C++ symbols."""
        nodes = []
        imports = []
        
        def visit(node, namespace=""):
            if node.type == 'preproc_include':
                imports.append(self._get_text(node, content))
            
            elif node.type == 'function_definition':
                # Try to get function name from declarator
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    name = self._extract_function_name_cpp(declarator, content)
                    if name:
                        full_name = f"{namespace}::{name}" if namespace else name
                        code = self._get_text(node, content)
                        calls = self._extract_calls_cpp(node, content)
                        api_calls = self._extract_api_calls(code)
                        comment_meta = self._extract_symbol_docs(node, content, code, 'other')
                        
                        nodes.append({
                            "name": full_name,
                            "type": "function",
                            "code": code,
                            "calls": calls,
                            "api_route": None,
                            "api_outbound": api_calls,
                            **comment_meta
                        })
            
            elif node.type == 'class_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = self._get_text(name_node, content)
                    body = node.child_by_field_name('body')
                    if body:
                        for child in body.children:
                            visit(child, class_name)
                    return
            
            for child in node.children:
                visit(child, namespace)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": "\n".join(imports)}

    def _extract_function_name_cpp(self, declarator, content: str) -> Optional[str]:
        """Extract function name from C++ declarator."""
        if declarator.type == 'function_declarator':
            decl = declarator.child_by_field_name('declarator')
            if decl:
                if decl.type == 'identifier':
                    return self._get_text(decl, content)
                elif decl.type == 'field_identifier':
                    return self._get_text(decl, content)
                elif decl.type == 'qualified_identifier':
                    name = decl.child_by_field_name('name')
                    if name:
                        return self._get_text(name, content)
        elif declarator.type == 'identifier':
            return self._get_text(declarator, content)
        return None

    def _extract_calls_cpp(self, node, content: str) -> List[str]:
        """Extract function calls from C/C++ code."""
        calls = set()
        
        def visit(n):
            if n.type == 'call_expression':
                func = n.child_by_field_name('function')
                if func:
                    if func.type == 'identifier':
                        calls.add(self._get_text(func, content))
                    elif func.type == 'field_expression':
                        field = func.child_by_field_name('field')
                        if field:
                            calls.add(self._get_text(field, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_bash(self, root, content: str) -> Dict[str, Any]:
        """Extract Bash/Shell script symbols."""
        nodes = []
        imports = []
        
        def visit(node):
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_text(name_node, content)
                    code = self._get_text(node, content)
                    calls = self._extract_calls_bash(node, content)
                    
                    comment_meta = self._extract_symbol_docs(node, content, code, 'other')
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": [],
                        **comment_meta
                    })
            
            for child in node.children:
                visit(child)
        
        visit(root)
        return {"nodes": nodes, "imports": imports, "globals": ""}

    def _extract_calls_bash(self, node, content: str) -> List[str]:
        """Extract function calls from Bash script."""
        calls = set()
        
        def visit(n):
            if n.type == 'command':
                name = n.child_by_field_name('name')
                if name and name.type == 'command_name':
                    first_child = name.children[0] if name.children else None
                    if first_child:
                        calls.add(self._get_text(first_child, content))
            
            for child in n.children:
                visit(child)
        
        visit(node)
        return list(calls)

    def _extract_api_calls(self, code_snippet: str) -> List[str]:
        """Extract API endpoint calls from code."""
        endpoints = []
        for pattern in self.API_CALL_PATTERNS:
            matches = re.findall(pattern, code_snippet)
            endpoints.extend([normalize_route(m) for m in matches])
        return list(set(endpoints))

    def _parse_regex_fallback(self, content: str, ext: str, filename: str) -> Dict[str, Any]:
        """Enhanced regex fallback parser."""
        nodes = []
        
        # Determine language key
        lang_key = ext.replace('.', '')
        if lang_key == 'ino':
            lang_key = 'cpp'  # Arduino uses C++
        
        patterns = self.REGEX_PATTERNS.get(lang_key, [
            r'function\s+(\w+)\s*\(',
            r'def\s+(\w+)\s*\(',
            r'fn\s+(\w+)\s*\(',
        ])
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # Get the last captured group (the function name)
                    groups = match.groups()
                    name = groups[-1] if groups else match.group(1)
                    
                    code_snippet = "\n".join(lines[i:min(i+20, len(lines))])
                    
                    # Extract calls using basic patterns
                    call_pattern = r'(\w+)\s*\('
                    calls = list(set(re.findall(call_pattern, code_snippet)))
                    # Filter out common keywords
                    excludes = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'void', 'int', 'float'}
                    calls = [c for c in calls if c not in excludes]
                    
                    api_calls = self._extract_api_calls(code_snippet)
                    
                    docstring = _extract_block_docstring(code_snippet)
                    comment_tokens = len(re.findall(r"[a-zA-Z_]+", docstring))
                    code_tokens = max(len(re.findall(r"[a-zA-Z_]+", code_snippet)), 1)
                    nodes.append({
                        "name": name,
                        "type": "function",
                        "code": code_snippet,
                        "calls": calls,
                        "api_route": None,
                        "api_outbound": api_calls,
                        "docstring": docstring,
                        "comments_above": "",
                        "comment_ratio": comment_tokens / code_tokens,
                        "comment_quality": _comment_quality_score(docstring, "")
                    })
                    break
        
        return {"nodes": nodes, "imports": [], "globals": ""}

# --- 3. Project Summarization System ---

class ProjectSummarizer:
    """
    Generates high-level summaries of files and the overall project architecture.
    Uses LLM to synthesize information for broader context.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.file_summaries = {}
        self.folder_summaries = {}
        self.project_summary = ""

    async def summarize_file(self, filename: str, content: str, nodes: List[Dict]) -> str:
        """Generate a concise summary of a single file."""
        node_names = [n['name'] for n in nodes[:10]]
        node_str = ", ".join(node_names)
        
        prompt = f"""
Summarize the following file in 1-2 concise sentences. 
Focus on its primary responsibility in the system.
Filename: {filename}
Key Symbols: {node_str}
Content Preview: {content[:1000]}
"""
        try:
            res = await safe_chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = res.choices[0].message.content.strip()
            self.file_summaries[filename] = summary
            return summary
        except Exception as e:
            print(f"   ⚠️ Summary failed for {filename}: {e}")
            return f"Module containing {node_str}"

    async def build_hierarchical_summaries(self, repo_name: str, files_data: Dict[str, Dict]):
        """Build folder-level summaries from file summaries (hierarchical map)."""
        folder_to_files = defaultdict(list)
        for filename in files_data.keys():
            folder = os.path.dirname(filename) or "."
            folder_to_files[folder].append(filename)

        # Bottom-up summarization by folder depth.
        for folder in sorted(folder_to_files.keys(), key=lambda f: f.count('/'), reverse=True):
            file_summaries = [self.file_summaries.get(f, "Source file") for f in folder_to_files[folder][:20]]
            child_folders = [f for f in folder_to_files.keys() if f != folder and f.startswith(folder + "/") and f.count('/') == folder.count('/') + 1]
            child_summaries = [self.folder_summaries.get(f"{repo_name}:{cf}", "") for cf in child_folders[:10]]

            composed = "\n".join([f"- {s}" for s in file_summaries + child_summaries if s])
            if not composed:
                composed = "- Source code folder"

            prompt = f"""
Summarize this folder in 1-2 concise sentences.
Repository: {repo_name}
Folder: {folder}
Contained Summaries:
{composed[:2500]}
"""
            try:
                res = await safe_chat_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                self.folder_summaries[f"{repo_name}:{folder}"] = res.choices[0].message.content.strip()
            except Exception:
                self.folder_summaries[f"{repo_name}:{folder}"] = f"Folder containing {len(folder_to_files[folder])} files"

    async def generate_project_overview(self, multi_graph: Dict[str, Dict]) -> str:
        """Synthesize project-level architecture summary."""
        print("🧠 Generating project architectural overview...", flush=True)
        
        file_overviews = []
        for repo, graph in multi_graph.items():
            repo_files = set()
            for node_id, data in graph.items():
                repo_files.add(data['file'])
            
            # Use a larger subset for the overview to be more comprehensive
            subset = list(repo_files)[:100]
            summaries = [f"- {f}: {self.file_summaries.get(f, 'Source file')}" for f in subset]
            if len(repo_files) > 100:
                summaries.append(f"- ... and {len(repo_files) - 100} more files.")
            file_overviews.append(f"Repo [{repo}]:\n" + "\n".join(summaries))
        
        all_files_str = "\n\n".join(file_overviews)
        
        prompt = f"""
Given the following list of files and their short descriptions, provide a high-level architectural overview of the project.
Describe the main components, how they interact, and the overall purpose of the system.
Keep it under 3-4 paragraphs.

{all_files_str}
"""
        try:
            res = await safe_chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            self.project_summary = res.choices[0].message.content.strip()
            return self.project_summary
        except Exception as e:
            print(f"   ⚠️ Project overview failed: {e}")
            return "Multi-repository software project."

    def save(self, filepath: str):
        """Save summaries to disk."""
        data = {
            "file_summaries": self.file_summaries,
            "folder_summaries": self.folder_summaries,
            "project_summary": self.project_summary
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load summaries from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.file_summaries = data.get("file_summaries", {})
                self.folder_summaries = data.get("folder_summaries", {})
                self.project_summary = data.get("project_summary", "")

# --- 4. Import Resolution System ---

class ImportResolver:
    """Resolves imports to enable namespace-aware linking."""
    
    def __init__(self, repo_files: Dict[str, Dict]):
        self.repo_files = repo_files
        self.import_map = {}
        self.export_map = {}
        
    def build_maps(self, parsed_data: Dict[str, Dict]):
        """Build import and export maps for the repository."""
        
        # First pass: collect exports
        for filepath, data in parsed_data.items():
            exports = set()
            for node in data.get('nodes', []):
                exports.add(node['name'].split('.')[-1])
            self.export_map[filepath] = exports
        
        # Second pass: resolve imports
        for filepath, data in parsed_data.items():
            self.import_map[filepath] = {}
            
            for import_stmt in data.get('imports', []):
                resolved = self._resolve_import(filepath, import_stmt)
                if resolved:
                    for symbol, source in resolved.items():
                        self.import_map[filepath][symbol] = source
    
    def _resolve_import(self, current_file: str, import_stmt: str) -> Dict[str, str]:
        """Resolve a single import statement to source file."""
        result = {}
        
        # Python
        py_match = re.search(r'from\s+([\w.]+)\s+import\s+(.+)', import_stmt)
        if py_match:
            module = py_match.group(1)
            imports = [s.strip() for s in py_match.group(2).split(',')]
            
            potential_paths = [
                f"{module.replace('.', '/')}.py",
                f"{module.replace('.', '/')}/__init__.py"
            ]
            
            for path in potential_paths:
                if path in self.repo_files:
                    for sym in imports:
                        result[sym.split(' as ')[-1].strip()] = path
                    break
        
        # JavaScript/TypeScript
        js_match = re.search(r'import\s+\{([^}]+)\}\s+from\s+["\']([^"\']+)["\']', import_stmt)
        if js_match:
            imports = [s.strip() for s in js_match.group(1).split(',')]
            path = js_match.group(2)
            
            resolved_path = self._resolve_relative_path(current_file, path)
            if resolved_path and resolved_path in self.repo_files:
                for sym in imports:
                    result[sym.split(' as ')[-1].strip()] = resolved_path
        
        js_default = re.search(r'import\s+(\w+)\s+from\s+["\']([^"\']+)["\']', import_stmt)
        if js_default:
            symbol = js_default.group(1)
            path = js_default.group(2)
            resolved_path = self._resolve_relative_path(current_file, path)
            if resolved_path and resolved_path in self.repo_files:
                result[symbol] = resolved_path
        
        # C/C++ includes
        cpp_match = re.search(r'#include\s+["\']([^"\']+)["\']', import_stmt)
        if cpp_match:
            include_file = cpp_match.group(1)
            resolved_path = self._resolve_relative_path(current_file, include_file)
            if resolved_path and resolved_path in self.repo_files:
                # For C/C++, we don't know specific symbols, so return empty
                pass
        
        return result
    
    def _resolve_relative_path(self, current_file: str, import_path: str) -> Optional[str]:
        """Resolve relative import path to actual file path."""
        if not import_path.startswith('.'):
            return None
        
        current_dir = os.path.dirname(current_file)
        resolved = os.path.normpath(os.path.join(current_dir, import_path))
        
        # Try with different extensions
        extensions = ['.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.ts', '.h', '.hpp']
        for ext in extensions:
            candidate = resolved + ext
            if candidate in self.repo_files:
                return candidate
        
        # Try exact match
        if resolved in self.repo_files:
            return resolved
        
        return None
    
    def resolve_call(self, filepath: str, symbol: str) -> Optional[str]:
        """Resolve a symbol call to its source file."""
        return self.import_map.get(filepath, {}).get(symbol)

# --- 4. Vector Embedding System ---

class VectorEmbeddingStore:
    """Stores and searches function embeddings using FAISS + sparse hybrid scoring."""

    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.intent_index = None
        self.code_index = None
        self.structure_index = None
        self.node_ids = []
        self.node_metadata = {}
        self.last_scores = {}
        self.embedding_cache = {"intent": {}, "code": {}, "structure": {}}

        # Sparse retrieval state (BM25-like)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.df = defaultdict(int)
        self.avg_doc_len = 1.0

        if HAS_FAISS:
            self._reset_indices()

    def _reset_indices(self):
        if HAS_FAISS:
            self.intent_index = faiss.IndexFlatIP(self.dimension)
            self.code_index = faiss.IndexFlatIP(self.dimension)
            self.structure_index = faiss.IndexFlatIP(self.dimension)

    def _compose_node_texts(self, node_id: str, data: Dict[str, Any]) -> Tuple[str, str, str]:
        code_preview = data.get('code', '')[:1200]
        symbol_name = data.get('symbol', {}).get('name', node_id.split('::')[-1])

        intent_text = f"""
Function: {symbol_name}
File: {data.get('file', '')}
Route: {data.get('api_route') or ''}
Docstring:
{data.get('docstring', '')}
Developer Comments:
{data.get('comments_above', '')}
"""
        code_text = f"""
Function: {symbol_name}
File: {data.get('file', '')}
Type: {data.get('type', 'function')}
Code:
{code_preview}
"""
        structure_text = f"""
Function: {symbol_name}
Type: {data.get('type', 'function')}
Args: {', '.join(data.get('arg_names', []))}
Arg Types: {', '.join(data.get('arg_types', []))}
Return Type: {data.get('return_type') or ''}
Decorators: {', '.join(data.get('decorators', []))}
Exceptions: {', '.join(data.get('exception_types', []))}
Inherits: {', '.join(data.get('bases', []))}
"""
        return intent_text, code_text, structure_text

    def _build_sparse_index(self, texts: List[str]):
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.df = defaultdict(int)

        for text in texts:
            toks = _tokenize_for_sparse(text)
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            self.doc_term_freqs.append(dict(tf))
            self.doc_lengths.append(len(toks))
            for t in set(toks):
                self.df[t] += 1

        self.avg_doc_len = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 1.0

    def _bm25_scores(self, query: str) -> Dict[str, float]:
        if not self.node_ids or not self.doc_term_freqs:
            return {}
        q_tokens = _tokenize_for_sparse(query)
        if not q_tokens:
            return {}

        N = len(self.node_ids)
        k1, b = 1.5, 0.75
        scores = defaultdict(float)

        for q in q_tokens:
            df = self.df.get(q, 0)
            if df == 0:
                continue
            idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
            for idx, tf_map in enumerate(self.doc_term_freqs):
                tf = tf_map.get(q, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx] if idx < len(self.doc_lengths) else self.avg_doc_len
                denom = tf + k1 * (1 - b + b * (dl / max(self.avg_doc_len, 1e-6)))
                scores[self.node_ids[idx]] += idf * (tf * (k1 + 1)) / max(denom, 1e-6)

        return dict(scores)

    async def build_index(self, graph: Dict[str, Dict]):
        """Build intent/code/structure FAISS indices and sparse lexical stats."""
        await self.update_index(graph)

    async def update_index(
        self,
        graph: Dict[str, Dict],
        old_graph_ids: Optional[Set[str]] = None,
        previous_store: Optional['VectorEmbeddingStore'] = None
    ):
        """Incrementally refresh index by re-embedding only new/changed nodes, while rebuilding FAISS structure."""
        if not HAS_FAISS or not graph:
            return

        old_graph_ids = old_graph_ids or set()
        prev_intent = {}
        prev_code = {}
        prev_structure = {}
        if previous_store:
            prev_intent = dict(previous_store.embedding_cache.get('intent', {}))
            prev_code = dict(previous_store.embedding_cache.get('code', {}))
            prev_structure = dict(previous_store.embedding_cache.get('structure', {}))

        node_ids = list(graph.keys())
        intent_texts, code_texts, structure_texts = [], [], []
        ids_to_embed = []

        for node_id in node_ids:
            data = graph[node_id]
            self.node_metadata[node_id] = data
            intent_text, code_text, structure_text = self._compose_node_texts(node_id, data)
            intent_texts.append(intent_text)
            code_texts.append(code_text)
            structure_texts.append(structure_text)

            if (
                node_id in old_graph_ids
                and node_id in prev_intent
                and node_id in prev_code
                and node_id in prev_structure
            ):
                continue
            ids_to_embed.append(node_id)

        self.embedding_cache = {"intent": {}, "code": {}, "structure": {}}
        for node_id in node_ids:
            if node_id in prev_intent and node_id in prev_code and node_id in prev_structure and node_id in old_graph_ids:
                self.embedding_cache['intent'][node_id] = prev_intent[node_id]
                self.embedding_cache['code'][node_id] = prev_code[node_id]
                self.embedding_cache['structure'][node_id] = prev_structure[node_id]

        if ids_to_embed:
            for i in range(0, len(ids_to_embed), EMBEDDING_BATCH_SIZE):
                id_batch = ids_to_embed[i:i + EMBEDDING_BATCH_SIZE]
                i_batch, c_batch, s_batch = [], [], []
                for nid in id_batch:
                    data = graph[nid]
                    intent_text, code_text, structure_text = self._compose_node_texts(nid, data)
                    i_batch.append(intent_text)
                    c_batch.append(code_text)
                    s_batch.append(structure_text)

                intent_embeddings = await get_embeddings_batch(i_batch)
                code_embeddings = await get_embeddings_batch(c_batch)
                structure_embeddings = await get_embeddings_batch(s_batch)

                for idx, nid in enumerate(id_batch):
                    self.embedding_cache['intent'][nid] = intent_embeddings[idx]
                    self.embedding_cache['code'][nid] = code_embeddings[idx]
                    self.embedding_cache['structure'][nid] = structure_embeddings[idx]

                print(f"      Embedded {min(i + EMBEDDING_BATCH_SIZE, len(ids_to_embed))}/{len(ids_to_embed)} changed nodes", flush=True)

        self._build_sparse_index([f"{intent_texts[i]}\n{code_texts[i]}\n{structure_texts[i]}" for i in range(len(node_ids))])

        self._reset_indices()
        ordered_intent = [self.embedding_cache['intent'][nid] for nid in node_ids if nid in self.embedding_cache['intent']]
        ordered_code = [self.embedding_cache['code'][nid] for nid in node_ids if nid in self.embedding_cache['code']]
        ordered_structure = [self.embedding_cache['structure'][nid] for nid in node_ids if nid in self.embedding_cache['structure']]

        if ordered_intent and ordered_code and ordered_structure:
            intent_arr = np.array(ordered_intent, dtype=np.float32)
            code_arr = np.array(ordered_code, dtype=np.float32)
            structure_arr = np.array(ordered_structure, dtype=np.float32)
            faiss.normalize_L2(intent_arr)
            faiss.normalize_L2(code_arr)
            faiss.normalize_L2(structure_arr)
            self.intent_index.add(intent_arr)
            self.code_index.add(code_arr)
            self.structure_index.add(structure_arr)
            self.node_ids = [nid for nid in node_ids if nid in self.embedding_cache['intent']]

    async def search(self, query: str, k: int = TOP_K_SEEDS) -> List[str]:
        """Hybrid search: embedding + BM25 + centrality."""
        if not HAS_FAISS or self.intent_index is None or self.code_index is None or self.structure_index is None or self.intent_index.ntotal == 0:
            return []

        query_emb = await get_embeddings_batch([query])
        if not query_emb:
            return []

        query_array = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(query_array)

        k = min(k, len(self.node_ids))
        if k == 0:
            return []

        probe_k = min(len(self.node_ids), max(k * 5, 40))
        i_dist, i_idx = self.intent_index.search(query_array, probe_k)
        c_dist, c_idx = self.code_index.search(query_array, probe_k)
        s_dist, s_idx = self.structure_index.search(query_array, probe_k)

        intent_scores, code_scores, struct_scores = {}, {}, {}
        for rank, idx in enumerate(i_idx[0]):
            if idx < len(self.node_ids):
                intent_scores[self.node_ids[idx]] = float(i_dist[0][rank])
        for rank, idx in enumerate(c_idx[0]):
            if idx < len(self.node_ids):
                code_scores[self.node_ids[idx]] = float(c_dist[0][rank])
        for rank, idx in enumerate(s_idx[0]):
            if idx < len(self.node_ids):
                struct_scores[self.node_ids[idx]] = float(s_dist[0][rank])

        bm25_raw = self._bm25_scores(query)
        bm25_scores = _normalize_scores(bm25_raw)
        candidate_ids = set(intent_scores) | set(code_scores) | set(struct_scores) | set(bm25_scores)

        candidates = []
        for node_id in candidate_ids:
            metadata = self.node_metadata.get(node_id, {})
            intent_similarity = intent_scores.get(node_id, 0.0)
            code_similarity = code_scores.get(node_id, 0.0)
            structure_similarity = struct_scores.get(node_id, 0.0)
            embedding_score = (0.6 * intent_similarity) + (0.25 * code_similarity) + (0.15 * structure_similarity)
            bm25_score = bm25_scores.get(node_id, 0.0)
            centrality_score = float(metadata.get('centrality_score', 0.0))
            
            # --- NEW: Direct Name Match Boost ---
            symbol_name = metadata.get('symbol', {}).get('name', '')
            name_match_boost = 0.0
            
            # If the query *is* the function name (exact match)
            if query.strip() == symbol_name:
                name_match_boost = 5.0  # Massive boost to force to top
            
            # If the query contains the function name (e.g. "what does processPayment do")
            elif symbol_name and len(symbol_name) > 3 and symbol_name in query:
                name_match_boost = 2.0
            # -------------------------------------

            comments_blob = f"{metadata.get('docstring', '')}\n{metadata.get('comments_above', '')}"
            comment_bonus = _comment_keyword_overlap_score(query, comments_blob)
            
            final_score = (
                (EMBEDDING_WEIGHT * embedding_score)
                + (BM25_WEIGHT * bm25_score)
                + (CENTRALITY_WEIGHT * centrality_score)
                + (COMMENT_MATCH_WEIGHT * comment_bonus)
                + name_match_boost  # <--- Add the boost here
            )
            candidates.append((final_score, node_id))

        candidates.sort(reverse=True, key=lambda item: item[0])
        candidates = _apply_graph_distance_reranking(candidates[: max(k * 10, 20)], self.node_metadata)
        self.last_scores = {node_id: score for score, node_id in candidates}
        return [node_id for score, node_id in candidates[:k]]

    def save(self, filepath: str):
        """Save index to disk."""
        if HAS_FAISS and self.intent_index and self.code_index and self.structure_index and self.intent_index.ntotal > 0:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            faiss.write_index(self.intent_index, filepath)
            faiss.write_index(self.code_index, filepath + '.code')
            faiss.write_index(self.structure_index, filepath + '.struct')

            with open(filepath + '.meta', 'w') as f:
                json.dump({
                    'node_ids': self.node_ids,
                    'node_metadata': self.node_metadata,
                    'doc_term_freqs': self.doc_term_freqs,
                    'doc_lengths': self.doc_lengths,
                    'df': dict(self.df),
                    'avg_doc_len': self.avg_doc_len,
                    'embedding_cache': self.embedding_cache
                }, f)
        else:
            raise RuntimeError("Cannot save index: FAISS unavailable or index is empty.")

    def load(self, filepath: str):
        """Load index from disk."""
        if HAS_FAISS and os.path.exists(filepath):
            self.intent_index = faiss.read_index(filepath)
            self.code_index = faiss.read_index(filepath + '.code') if os.path.exists(filepath + '.code') else self.intent_index
            self.structure_index = faiss.read_index(filepath + '.struct') if os.path.exists(filepath + '.struct') else self.intent_index
            with open(filepath + '.meta', 'r') as f:
                data = json.load(f)
                self.node_ids = data.get('node_ids', [])
                self.node_metadata = data.get('node_metadata', {})
                self.doc_term_freqs = data.get('doc_term_freqs', [])
                self.doc_lengths = data.get('doc_lengths', [])
                self.df = defaultdict(int, data.get('df', {}))
                self.avg_doc_len = data.get('avg_doc_len', 1.0)
                self.embedding_cache = data.get('embedding_cache', {"intent": {}, "code": {}, "structure": {}})


# --- Graph Building ---

async def build_single_repo_graph(
    repo_name: str,
    files_data: Dict[str, Dict],
    previous_graph: Optional[Dict[str, Dict]] = None,
    changed_files: Optional[Set[str]] = None
) -> Tuple[Dict, 'ImportResolver']:
    """Build graph with incremental Tree-sitter parsing and import resolution."""
    parser = TreeSitterParser()

    if changed_files is None:
        changed_files = set(files_data.keys())
    else:
        changed_files = set(changed_files)

    removed_files = set()
    if previous_graph:
        removed_files = {n.get('file') for n in previous_graph.values() if n.get('file') and n.get('file') not in files_data}

    files_to_reparse = {fname for fname in changed_files if fname in files_data}
    files_to_purge = files_to_reparse | removed_files

    graph = {}
    if previous_graph:
        for node_id, node in previous_graph.items():
            node_file = node.get('file')
            node_type = node.get('type')
            if node_type == 'folder':
                continue
            if node_file in files_to_purge:
                continue
            copied = dict(node)
            copied['dependencies'] = list(node.get('dependencies', []))
            copied['inherits'] = list(node.get('inherits', []))
            copied['callers'] = []
            copied['cross_repo_deps'] = list(node.get('cross_repo_deps', []))
            graph[node_id] = copied

    parsed_data = {}
    print(f"   🔍 Parsing {len(files_to_reparse)} changed files in [{repo_name}]...")
    parse_semaphore = asyncio.Semaphore(16)

    async def _parse_one(filename: str, data: Dict[str, str]):
        async with parse_semaphore:
            result = await asyncio.to_thread(parser.parse, filename, data['content'])
            return filename, result

    parse_results = await asyncio.gather(*[_parse_one(filename, files_data[filename]) for filename in files_to_reparse]) if files_to_reparse else []
    for filename, result in parse_results:
        parsed_data[filename] = result

    resolver = ImportResolver(files_data)
    resolver.build_maps(parsed_data)

    symbol_registry = defaultdict(list)
    for node_id, ndata in graph.items():
        symbol_name = ndata.get('symbol', {}).get('name', node_id.split('::')[-1])
        symbol_registry[symbol_name].append({
            "file": ndata.get('file', ''),
            "symbol": symbol_name,
            "type": ndata.get('type', 'function'),
            "code": ndata.get('code', ''),
            "calls": [],
            "api_route": ndata.get('api_route'),
            "api_outbound": ndata.get('api_outbound', []),
            "docstring": ndata.get('docstring', ''),
            "comments_above": ndata.get('comments_above', ''),
            "comment_ratio": ndata.get('comment_ratio', 0.0),
            "comment_quality": ndata.get('comment_quality', 0.0),
            "arg_names": ndata.get('arg_names', []),
            "arg_types": ndata.get('arg_types', []),
            "return_type": ndata.get('return_type'),
            "variable_mentions": ndata.get('variable_mentions', []),
            "exception_types": ndata.get('exception_types', []),
            "decorators": ndata.get('decorators', []),
            "bases": ndata.get('bases', [])
        })

    file_globals = {}
    for filename, result in parsed_data.items():
        file_globals[filename] = result['globals']
        for node in result['nodes']:
            name = node['name']
            symbol_registry[name].append({
                "file": filename,
                "symbol": name,
                "type": node['type'],
                "code": node['code'],
                "calls": node['calls'],
                "api_route": node.get('api_route'),
                "api_outbound": node.get('api_outbound', []),
                "docstring": node.get('docstring', ''),
                "comments_above": node.get('comments_above', ''),
                "comment_ratio": node.get('comment_ratio', 0.0),
                "comment_quality": node.get('comment_quality', 0.0),
                "arg_names": node.get('arg_names', []),
                "arg_types": node.get('arg_types', []),
                "return_type": node.get('return_type'),
                "variable_mentions": node.get('variable_mentions', []),
                "exception_types": node.get('exception_types', []),
                "decorators": node.get('decorators', []),
                "bases": node.get('bases', [])
            })

    defined_symbols = set(symbol_registry.keys())

    for filename, result in parsed_data.items():
        for node in result['nodes']:
            sym_name = node['name']
            node_id = f"{filename}::{sym_name}"
            valid_deps = []

            for called_func in node.get('calls', []):
                source_file = resolver.resolve_call(filename, called_func)
                if source_file:
                    candidates = symbol_registry.get(called_func, [])
                    for candidate in candidates:
                        if candidate['file'] == source_file:
                            target_id = f"{candidate['file']}::{called_func}"
                            if target_id != node_id:
                                valid_deps.append(target_id)
                            break
                elif called_func in defined_symbols:
                    for target in symbol_registry[called_func]:
                        target_id = f"{target['file']}::{called_func}"
                        if target_id != node_id:
                            valid_deps.append(target_id)

            node_dfg = _extract_python_data_flow(node['code']) if filename.endswith('.py') else {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}}
            node_cfg = _extract_python_cfg(node['code']) if filename.endswith('.py') else {"blocks": [], "edges": []}
            node_types = _extract_type_awareness(node['code']) if filename.endswith('.py') else {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}}

            graph[node_id] = {
                "repo": repo_name,
                "symbol": {"repo": repo_name, "file": filename, "name": sym_name},
                "file": filename,
                "code": node['code'],
                "type": node['type'],
                "globals": file_globals.get(filename, ""),
                "dependencies": list(set(valid_deps)),
                "inherits": [],
                "callers": [],
                "api_route": node.get('api_route'),
                "api_outbound": node.get('api_outbound', []),
                "docstring": node.get('docstring', ''),
                "comments_above": node.get('comments_above', ''),
                "comment_ratio": node.get('comment_ratio', 0.0),
                "comment_quality": node.get('comment_quality', 0.0),
                "arg_names": node.get('arg_names', []),
                "arg_types": node.get('arg_types', []),
                "return_type": node.get('return_type'),
                "variable_mentions": node.get('variable_mentions', []),
                "exception_types": node.get('exception_types', []),
                "decorators": node.get('decorators', []),
                "bases": node.get('bases', []),
                "global_id": _global_node_id(repo_name, filename, sym_name),
                "node_hash": "",
                "pagerank": 0.0,
                "centrality_score": 0.0,
                "cross_repo_deps": [],
                "dfg": node_dfg,
                "cfg": node_cfg,
                "type_info": node_types,
                "param_bindings": _extract_param_bindings(node['code']) if filename.endswith('.py') else {},
                "symbolic_execution": _symbolic_execution_preview(node['code']) if filename.endswith('.py') else {"constraints": [], "symbols": []},
                "test_inputs": _generate_boundary_test_inputs(node.get('arg_names', []), node.get('arg_types', [])),
                "security_role": "",
                "requires_auth": any('auth' in (d or '').lower() or 'login_required' in (d or '').lower() for d in node.get('decorators', [])),
                "previous_identity": None
            }

    for filename in files_to_reparse:
        file_has_nodes = any(node.get('file') == filename for node in graph.values())
        if not file_has_nodes:
            node_id = f"{filename}::module"
            graph[node_id] = {
                "repo": repo_name,
                "symbol": {"repo": repo_name, "file": filename, "name": "module"},
                "file": filename,
                "code": files_data[filename]['content'][:2000],
                "type": "module",
                "globals": file_globals.get(filename, ""),
                "dependencies": [],
                "inherits": [],
                "callers": [],
                "api_route": None,
                "api_outbound": [],
                "docstring": "",
                "comments_above": "",
                "comment_ratio": 0.0,
                "comment_quality": 0.0,
                "arg_names": [],
                "arg_types": [],
                "return_type": None,
                "variable_mentions": [],
                "exception_types": [],
                "decorators": [],
                "bases": [],
                "global_id": _global_node_id(repo_name, filename, "module"),
                "node_hash": "",
                "pagerank": 0.0,
                "centrality_score": 0.0,
                "cross_repo_deps": [],
                "dfg": {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}},
                "cfg": {"blocks": [], "edges": []},
                "type_info": {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}},
                "param_bindings": {},
                "security_role": "neutral",
                "requires_auth": False,
                "previous_identity": None
            }

    for filename in files_data.keys():
        file_has_nodes = any(node.get('file') == filename for node in graph.values())
        if not file_has_nodes:
            node_id = f"{filename}::module"
            graph[node_id] = {
                "repo": repo_name,
                "symbol": {"repo": repo_name, "file": filename, "name": "module"},
                "file": filename,
                "code": files_data[filename]['content'][:2000],
                "type": "module",
                "globals": "",
                "dependencies": [],
                "inherits": [],
                "callers": [],
                "api_route": None,
                "api_outbound": [],
                "docstring": "",
                "comments_above": "",
                "comment_ratio": 0.0,
                "comment_quality": 0.0,
                "arg_names": [],
                "arg_types": [],
                "return_type": None,
                "variable_mentions": [],
                "exception_types": [],
                "decorators": [],
                "bases": [],
                "global_id": _global_node_id(repo_name, filename, "module"),
                "node_hash": "",
                "pagerank": 0.0,
                "centrality_score": 0.0,
                "cross_repo_deps": [],
                "dfg": {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}},
                "cfg": {"blocks": [], "edges": []},
                "type_info": {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}},
                "param_bindings": {},
                "security_role": "neutral",
                "requires_auth": False,
                "previous_identity": None
            }

    folder_children = defaultdict(set)
    file_root_nodes = defaultdict(list)
    for nid, ndata in graph.items():
        file_root_nodes[ndata['file']].append(nid)

    for filename in files_data.keys():
        folder = os.path.dirname(filename) or "."
        current = folder
        prev = None
        while True:
            folder_key = f"folder::{current}"
            if prev is not None:
                folder_children[folder_key].add(f"folder::{prev}")
            prev = current
            if current in ("", "."):
                break
            current = os.path.dirname(current) or "."

        leaf_folder = f"folder::{folder}"
        for node_id in file_root_nodes.get(filename, []):
            folder_children[leaf_folder].add(node_id)

    for folder_key, children in folder_children.items():
        folder_path = folder_key.replace("folder::", "")
        folder_node_id = f"{folder_key}::node"
        graph[folder_node_id] = {
            "repo": repo_name,
            "symbol": {"repo": repo_name, "file": folder_path, "name": folder_key},
            "file": folder_path,
            "code": "",
            "type": "folder",
            "globals": "",
            "dependencies": [],
            "inherits": [],
            "callers": [],
            "contains": sorted(children),
            "api_route": None,
            "api_outbound": [],
            "docstring": "",
            "comments_above": "",
            "comment_ratio": 0.0,
            "comment_quality": 0.0,
            "arg_names": [],
            "arg_types": [],
            "return_type": None,
            "variable_mentions": [],
            "exception_types": [],
            "decorators": [],
            "bases": [],
            "global_id": _global_node_id(repo_name, folder_path, folder_key),
            "node_hash": "",
            "pagerank": 0.0,
            "centrality_score": 0.0,
            "cross_repo_deps": [],
            "dfg": {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}},
            "cfg": {"blocks": [], "edges": []},
            "type_info": {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}},
            "param_bindings": {},
            "symbolic_execution": {"constraints": [], "symbols": []},
            "test_inputs": [],
            "security_role": "neutral",
            "requires_auth": False,
            "previous_identity": None
        }

    class_name_to_ids = defaultdict(list)
    for nid, ndata in graph.items():
        if ndata.get('type') == 'class':
            class_name_to_ids[ndata.get('symbol', {}).get('name', '').split('.')[-1]].append(nid)

    for nid, ndata in graph.items():
        if ndata.get('type') != 'class':
            continue
        for base in ndata.get('bases', []):
            candidates = class_name_to_ids.get(base.split('.')[-1], [])
            if candidates:
                base_id = candidates[0]
                if base_id != nid and base_id not in ndata['inherits']:
                    ndata['inherits'].append(base_id)
                if base_id != nid and base_id not in ndata['dependencies']:
                    ndata['dependencies'].append(base_id)

    external_nodes = {}
    for filename, result in parsed_data.items():
        unresolved_libs = set()
        for imp_stmt in result.get('imports', []):
            lib = _extract_external_lib_from_import(imp_stmt)
            if not lib:
                continue
            if lib in {'from', 'import'}:
                continue
            if lib in [os.path.splitext(os.path.basename(f))[0] for f in files_data.keys()]:
                continue
            unresolved_libs.add(lib)

        file_nodes = [gid for gid, gdata in graph.items() if gdata.get('file') == filename]
        for fnid in file_nodes:
            graph[fnid]['dependencies'] = [d for d in graph[fnid].get('dependencies', []) if not str(d).startswith('EXT::')]

        for lib in unresolved_libs:
            ext_id = f"EXT::{lib}"
            if ext_id not in graph and ext_id not in external_nodes:
                external_nodes[ext_id] = {
                    "repo": repo_name,
                    "symbol": {"repo": repo_name, "file": "<external>", "name": lib},
                    "file": "<external>",
                    "code": f"external library: {lib}",
                    "type": "external_library",
                    "globals": "",
                    "dependencies": [],
                    "inherits": [],
                    "callers": [],
                    "api_route": None,
                    "api_outbound": [],
                    "docstring": f"External dependency: {lib}",
                    "comments_above": "",
                    "comment_ratio": 0.0,
                    "comment_quality": 0.0,
                    "arg_names": [], "arg_types": [], "return_type": None,
                    "variable_mentions": [], "exception_types": [], "decorators": [],
                    "bases": [],
                    "global_id": _global_node_id(repo_name, "<external>", lib),
                    "node_hash": "",
                    "pagerank": 0.0,
                    "centrality_score": 0.0,
                    "cross_repo_deps": []
                }
            for fnid in file_nodes:
                if ext_id not in graph[fnid]['dependencies']:
                    graph[fnid]['dependencies'].append(ext_id)

    graph.update(external_nodes)

    for nid, ndata in graph.items():
        ndata['dependencies'] = [dep for dep in ndata.get('dependencies', []) if dep in graph]
        ndata['callers'] = []

    for source_id, source_data in graph.items():
        for target_id in source_data.get('dependencies', []):
            if target_id in graph and source_id not in graph[target_id]['callers']:
                graph[target_id]['callers'].append(source_id)

    # Inter-procedural variable flow: map caller argument names -> callee params.
    for source_id, source_data in graph.items():
        for dep_id in source_data.get('dependencies', []):
            dep_node = graph.get(dep_id)
            if not dep_node:
                continue
            callee_params = dep_node.get('arg_names', [])
            caller_vars = []
            for v in source_data.get('dfg', {}).get('uses', {}).keys():
                if isinstance(v, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
                    caller_vars.append(v)
            bindings = {}
            for idx, param in enumerate(callee_params):
                if idx < len(caller_vars):
                    bindings[param] = caller_vars[idx]
            if bindings:
                dep_node.setdefault('param_bindings', {}).update(bindings)

    # Type/security enrichments and defaults
    for nid, ndata in graph.items():
        ndata.setdefault('dfg', {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}})
        ndata.setdefault('cfg', {"blocks": [], "edges": []})
        ndata.setdefault('type_info', {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}})
        ndata.setdefault('param_bindings', {})
        ndata.setdefault('previous_identity', None)
        ndata.setdefault('symbolic_execution', {"constraints": [], "symbols": []})
        ndata.setdefault('test_inputs', _generate_boundary_test_inputs(ndata.get('arg_names', []), ndata.get('arg_types', [])))
        ndata['security_role'] = _infer_security_role(ndata)
        if ndata.get('security_role') == 'auth_boundary':
            ndata['requires_auth'] = True

    # Add package-level dependency graph node.
    pkg_meta = _build_package_dependency_metadata(files_data)
    pkg_node_id = "package::dependencies"
    graph[pkg_node_id] = {
        "repo": repo_name,
        "symbol": {"repo": repo_name, "file": "package-manifest", "name": "package_dependencies"},
        "file": "package-manifest",
        "code": json.dumps(pkg_meta, ensure_ascii=False)[:4000],
        "type": "package_dependencies",
        "globals": "",
        "dependencies": [],
        "inherits": [],
        "callers": [],
        "api_route": None,
        "api_outbound": [],
        "docstring": "Dependency manifest extracted from package.json / requirements.txt",
        "comments_above": "",
        "comment_ratio": 0.0,
        "comment_quality": 0.0,
        "arg_names": [],
        "arg_types": [],
        "return_type": None,
        "variable_mentions": [],
        "exception_types": [],
        "decorators": [],
        "bases": [],
        "global_id": _global_node_id(repo_name, "package-manifest", "package_dependencies"),
        "node_hash": "",
        "pagerank": 0.0,
        "centrality_score": 0.0,
        "cross_repo_deps": [],
        "dfg": {"defs": {}, "uses": {}, "edges": [], "return_flows": [], "field_dependencies": {}, "taint": {"sources": [], "sinks": [], "sanitizers": [], "vulnerabilities": []}},
        "cfg": {"blocks": [], "edges": []},
        "type_info": {"type_registry": {}, "interface_links": [], "generic_usages": [], "field_links": {}},
        "param_bindings": {},
        "security_role": "neutral",
        "requires_auth": False,
        "previous_identity": None,
        "package_dependencies": pkg_meta
    }

    _compute_graph_centrality(graph)

    for nid, ndata in graph.items():
        ndata['node_hash'] = _compute_node_content_hash(ndata)

    _track_symbol_refactors(graph, previous_graph)

    print(f"   ✅ Built graph for [{repo_name}]: {len(graph)} nodes")
    return graph, resolver

def link_cross_repo_dependencies(multi_graph: Dict[str, Dict]) -> Dict[str, Dict]:
    """Link API calls across repositories."""
    
    route_map = defaultdict(list)
    
    for repo_name, graph in multi_graph.items():
        for node_id, data in graph.items():
            route = data.get('api_route')
            if route:
                normalized = normalize_route(route)
                route_map[normalized].append((repo_name, node_id))
    
    count_links = 0
    for repo_name, graph in multi_graph.items():
        for node_id, data in graph.items():
            outbound_routes = data.get('api_outbound', [])
            
            for route in outbound_routes:
                normalized = normalize_route(route)
                
                if normalized in route_map:
                    targets = route_map[normalized]
                    for target_repo, target_node_id in targets:
                        if target_repo == repo_name and target_node_id == node_id:
                            continue
                        
                        cross_ref = {
                            'repo': target_repo,
                            'node_id': target_node_id,
                            'route': route
                        }
                        
                        if cross_ref not in data['cross_repo_deps']:
                            data['cross_repo_deps'].append(cross_ref)
                            count_links += 1
    
    if count_links > 0:
        print(f"🌍 Linking cross-repo API dependencies...")
        print(f"   🔗 Established {count_links} cross-repo API connections")
    
    return multi_graph

def _detect_dependency_version_conflicts(multi_graph: Dict[str, Dict]) -> List[Dict[str, Any]]:
    seen = defaultdict(list)
    for repo, graph in multi_graph.items():
        pkg_node = graph.get('package::dependencies')
        if not pkg_node:
            continue
        for dep, ver in (pkg_node.get('package_dependencies', {}).get('all', {}) or {}).items():
            seen[dep].append((repo, str(ver)))

    conflicts = []
    for dep, vals in seen.items():
        versions = sorted(set(v for _, v in vals))
        if len(versions) > 1:
            conflicts.append({"dependency": dep, "versions": versions, "repos": vals})
    return conflicts

def _build_impact_tree_text(seeds, active_graph, depth=3):
    """
    Generates a structured map of what calls the target symbol, 
    now including Cross-Repository API consumers.
    """
    tree_lines = ["\n=== 💥 STRUCTURAL BLAST RADIUS (Callers & Victims) ==="]
    
    # 1. Pre-calculate an 'Incoming' cross-repo map for speed
    # This finds: "Which global_ids call WHICH other global_ids across repos?"
    incoming_cross_refs = defaultdict(list)
    for other_id, other_node in active_graph.items():
        for x_dep in other_node.get('cross_repo_deps', []):
            # The 'node_id' in cross_repo_deps is the target
            target_id = x_dep.get('node_id')
            incoming_cross_refs[target_id].append(other_id)

    for seed in seeds:
        node = active_graph.get(seed)
        if not node: continue
        
        symbol_name = node.get('symbol', {}).get('name', seed.split('::')[-1])
        tree_lines.append(f"TARGET: [{node.get('repo', 'UNKNOWN')}] {node['file']} -> {symbol_name}")
        
        def trace(nid, curr_depth, prefix=""):
            if curr_depth > depth: return
            
            target_node = active_graph.get(nid, {})
            # Combine: 
            # A) Standard local callers 
            # B) External repos calling this API (from our pre-calculated map)
            local_callers = target_node.get('callers', [])
            remote_callers = incoming_cross_refs.get(nid, [])
            all_callers = list(set(local_callers) | set(remote_callers))

            for i, caller_id in enumerate(all_callers):
                c_node = active_graph.get(caller_id)
                if not c_node: continue
                
                # Visual Formatting
                connector = "└── " if i == len(all_callers)-1 else "├── "
                repo_tag = f"[{c_node.get('repo', '???')}] "
                
                tree_lines.append(f"{prefix}{connector}{repo_tag}{c_node['file']}::{c_node['symbol']['name']}")
                
                # Recursively trace the 'victims of the victims'
                trace(caller_id, curr_depth + 1, prefix + ("    " if i == len(all_callers)-1 else "│   "))
        
        trace(seed, 1)
        
    return "\n".join(tree_lines)

async def build_multi_symbol_graph(
    multi_repo_data: Dict[str, Dict],
    summarizer: ProjectSummarizer,
    previous_graph: Optional[Dict[str, Dict]] = None,
    previous_vector_stores: Optional[Dict[str, 'VectorEmbeddingStore']] = None,
    previous_file_hashes: Optional[Dict[str, Dict[str, str]]] = None,
    current_file_hashes: Optional[Dict[str, Dict[str, str]]] = None
) -> Tuple[Dict, Dict]:
    """Build graphs for all repos with embeddings and summaries."""
    print("\n🕵️ Building symbol graphs with enhanced parsing...")

    multi_graph = {}
    resolvers = {}

    for repo_name, files_data in multi_repo_data.items():
        prev_repo_hashes = (previous_file_hashes or {}).get(repo_name, {})
        curr_repo_hashes = (current_file_hashes or {}).get(repo_name, {})
        changed_files = {
            fname for fname, fhash in curr_repo_hashes.items()
            if prev_repo_hashes.get(fname) != fhash
        }
        deleted_files = {fname for fname in prev_repo_hashes.keys() if fname not in curr_repo_hashes}
        changed_files.update(deleted_files)

        graph, resolver = await build_single_repo_graph(
            repo_name,
            files_data,
            previous_graph=(previous_graph or {}).get(repo_name, {}),
            changed_files=changed_files if previous_graph else None
        )

        print(f"   📝 Summarizing {len(files_data)} files in [{repo_name}]...")
        if changed_files:
            print(f"      ♻️ Partial re-index: {len(changed_files)} changed file(s) in [{repo_name}]")
        semaphore = asyncio.Semaphore(10)

        async def _summarize_one(fname: str, data: Dict[str, str]):
            async with semaphore:
                if changed_files and fname not in changed_files and fname in summarizer.file_summaries:
                    return
                symbol_names = [{"name": k.split("::")[-1]} for k in graph.keys() if graph[k]['file'] == fname]
                await summarizer.summarize_file(fname, data['content'], symbol_names)

        await asyncio.gather(*[_summarize_one(filename, data) for filename, data in files_data.items()])
        await summarizer.build_hierarchical_summaries(repo_name, files_data)

        multi_graph[repo_name] = graph
        resolvers[repo_name] = resolver

    multi_graph = link_cross_repo_dependencies(multi_graph)
    dep_conflicts = _detect_dependency_version_conflicts(multi_graph)
    if dep_conflicts:
        print(f"⚠️ Dependency version conflicts detected: {len(dep_conflicts)}")
        for c in dep_conflicts[:20]:
            print(f"   - {c['dependency']}: {', '.join(c['versions'])}")
            for repo, ver in c.get('repos', []):
                print(f"      · {repo}: {ver}")
    await summarizer.generate_project_overview(multi_graph)

    vector_stores = {}
    has_nodes = any(len(graph) > 0 for graph in multi_graph.values())

    if has_nodes and HAS_FAISS:
        print("   🧬 Building vector embeddings...")
        combined_graph = {}
        previous_vector_stores = previous_vector_stores or {}

        for repo_name, graph in multi_graph.items():
            if not graph:
                continue

            prev_repo_graph = (previous_graph or {}).get(repo_name, {})
            prev_store = previous_vector_stores.get(repo_name)
            old_graph_ids = set()
            if prev_repo_graph and prev_store:
                for old_id, old_node in prev_repo_graph.items():
                    new_node = graph.get(old_id)
                    if new_node and old_node.get('node_hash') == new_node.get('node_hash'):
                        old_graph_ids.add(old_id)

            store = VectorEmbeddingStore()
            await store.update_index(graph, old_graph_ids=old_graph_ids, previous_store=prev_store)
            vector_stores[repo_name] = store

            for node_id, node_data in graph.items():
                gid = node_data.get('global_id', _global_node_id(repo_name, node_data.get('file', ''), node_data.get('symbol', {}).get('name', node_id.split('::')[-1])))
                combined_graph[gid] = node_data

        if len(multi_graph) > 1 and combined_graph:
            global_candidates = combined_graph
            if len(combined_graph) > 7000:
                global_candidates = {
                    nid: nd for nid, nd in combined_graph.items()
                    if nd.get('api_route') or nd.get('cross_repo_deps') or nd.get('type') in {'external_library', 'folder'}
                }
            if global_candidates:
                global_store = VectorEmbeddingStore()
                await global_store.build_index(global_candidates)
                vector_stores['__global__'] = global_store

    with open(GRAPH_FILE, 'w') as f:
        json.dump(multi_graph, f, indent=2)

    try:
        export_graph_visualizations(multi_graph, VISUALIZATION_DIR)
    except Exception as vis_err:
        print(f"⚠️ Visualization export failed: {vis_err}")

    return multi_graph, vector_stores

# --- Enhanced Selector ---

def _extract_specific_targets(query: str, hints: List[str], active_graph: Dict) -> Set[str]:
    """Tiered lookup to ensure target symbols are never missed."""
    targets = set()
    # Extract potential symbol names from the query (CamelCase, snake_case, or dotted)
    query_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_\.]*', query)
    search_terms = set(hints or []) | set(query_tokens)

    for node_id, node_data in active_graph.items():
        symbol_name = node_data.get('symbol', {}).get('name', '')
        if not symbol_name: continue

        for term in search_terms:
            # TIER 1: Exact Match (Highest Priority)
            if symbol_name == term:
                targets.add(node_id)
            # TIER 2: Namespace/Script Match
            elif symbol_name.endswith(f".{term}") or term in node_data.get('file', ''):
                targets.add(node_id)
    return targets

def _node_expansion_priority(
    query: str,
    node: Dict[str, Any],
    distance: int,
    seed_score: float = 0.0,
    explanation_weight_mode: bool = False
) -> float:
    """Compute traversal priority score for a graph node."""
    comment_blob = f"{node.get('docstring', '')}\n{node.get('comments_above', '')}"
    comment_overlap = _comment_keyword_overlap_score(query, comment_blob)
    comment_density = min(float(node.get('comment_ratio', 0.0)), 1.0)
    comment_quality = float(node.get('comment_quality', 0.0))
    centrality = float(node.get('centrality_score', 0.0))
    graph_proximity = 1.0 / (distance + 1)

    comment_weight = 1.6 if explanation_weight_mode else 1.0
    quality_weight = 1.4 if explanation_weight_mode else 1.0

    return (
        seed_score
        + (comment_weight * comment_overlap)
        + (COMMENT_DENSITY_WEIGHT * comment_density)
        + (quality_weight * comment_quality)
        + (GRAPH_PROXIMITY_WEIGHT * graph_proximity)
        + (CENTRALITY_WEIGHT * centrality)
    )


def _priority_graph_traverse(
    seed_nodes: Set[str],
    active_graph: Dict,
    query: str,
    max_depth: int = BACKWARD_TRAVERSAL_DEPTH,
    direction: str = "dependencies",
    seed_scores: Optional[Dict[str, float]] = None,
    explanation_weight_mode: bool = False
) -> Set[str]:
    """Priority-based graph traversal that expands highest-value neighbors first."""
    import heapq

    seed_scores = seed_scores or {}
    visited_depth = {}
    selected = set()
    heap = []

    for node_id in seed_nodes:
        if node_id in active_graph:
            selected.add(node_id)
            visited_depth[node_id] = 0
            base_score = seed_scores.get(node_id, 0.0)
            priority = _node_expansion_priority(query, active_graph[node_id], 0, base_score, explanation_weight_mode)
            heapq.heappush(heap, (-priority, 0, node_id))

    edge_key = 'callers' if direction == 'callers' else 'dependencies'

    while heap:
        neg_priority, depth, node_id = heapq.heappop(heap)
        if depth >= max_depth:
            continue

        node = active_graph.get(node_id)
        if not node:
            continue

        for neighbor_id in node.get(edge_key, []):
            if neighbor_id not in active_graph:
                continue

            n_depth = depth + 1
            old_depth = visited_depth.get(neighbor_id)
            if old_depth is not None and old_depth <= n_depth:
                continue

            visited_depth[neighbor_id] = n_depth
            selected.add(neighbor_id)
            base_score = seed_scores.get(neighbor_id, 0.0)
            n_priority = _node_expansion_priority(query, active_graph[neighbor_id], n_depth, base_score, explanation_weight_mode)
            heapq.heappush(heap, (-n_priority, n_depth, neighbor_id))

    return selected


def _compute_graph_centrality(active_graph: Dict[str, Dict]):
    """Compute PageRank-based architectural importance."""
    if not HAS_NETWORKX or not active_graph:
        return

    g = nx.DiGraph()
    for nid, node in active_graph.items():
        g.add_node(nid)
        for dep in node.get('dependencies', []):
            if dep in active_graph:
                g.add_edge(nid, dep)

    try:
        pr = nx.pagerank(g, alpha=0.85)
        for nid, score in pr.items():
            if nid in active_graph:
                active_graph[nid]['pagerank'] = float(score)
                active_graph[nid]['centrality_score'] = min(float(score) * 50.0, 1.0)
    except Exception as e:
        print(f"⚠️ PageRank failed: {e}")


def _apply_graph_distance_reranking(candidates: List[Tuple[float, str]], active_graph: Dict[str, Dict]) -> List[Tuple[float, str]]:
    """Boost candidates that are topologically close to top anchor nodes."""
    if not candidates or not HAS_NETWORKX:
        return candidates

    graph = _build_nx_graph(active_graph)
    if graph is None:
        return candidates

    anchors = [nid for _, nid in sorted(candidates, reverse=True)[:3] if nid in graph]
    if not anchors:
        return candidates

    reranked = []
    for score, nid in candidates:
        if nid not in graph:
            reranked.append((score, nid))
            continue

        min_dist = 999
        for anchor in anchors:
            if anchor == nid:
                min_dist = 0
                break
            try:
                d = nx.shortest_path_length(graph, source=anchor, target=nid)
                if d < min_dist:
                    min_dist = d
            except Exception:
                continue

        graph_boost = 0.0 if min_dist == 999 else (1.0 / (min_dist + 1.0))
        reranked.append((score + (0.25 * graph_boost), nid))

    reranked.sort(reverse=True, key=lambda x: x[0])
    return reranked


def _get_blast_radius(target_node_id: str, active_graph: Dict[str, Dict], max_depth: int = 2) -> List[str]:
    """Reverse traversal impact report (who calls this node)."""
    impact_report = []
    visited = set()
    queue = [(target_node_id, 0)]
    impact_graph = defaultdict(list)

    while queue:
        curr, depth = queue.pop(0)
        if depth > max_depth or curr in visited:
            continue
        visited.add(curr)

        node = active_graph.get(curr)
        if not node:
            continue

        impact_graph[depth].append(f"{node.get('file')}::{node.get('symbol', {}).get('name')}")
        for caller in node.get('callers', []):
            if caller not in visited:
                queue.append((caller, depth + 1))

    impact_report.append(f"### 💥 IMPACT ANALYSIS: {target_node_id}")
    impact_report.append("Direct Callers (Immediate Breakage):")
    for caller in impact_graph.get(1, []):
        impact_report.append(f"- {caller}")

    impact_report.append("\nIndirect Impact (Downstream Ripple):")
    for depth in range(2, max_depth + 1):
        for caller in impact_graph.get(depth, []):
            impact_report.append(f"- (Depth {depth}) {caller}")

    return ["\n".join(impact_report)]


def _build_nx_graph(active_graph: Dict[str, Dict]) -> Optional[Any]:
    if not HAS_NETWORKX:
        return None
    g = nx.DiGraph()
    for node_id, node in active_graph.items():
        g.add_node(node_id)
        for dep in node.get('dependencies', []):
            if dep in active_graph:
                g.add_edge(node_id, dep)
        for caller in node.get('callers', []):
            if caller in active_graph:
                g.add_edge(caller, node_id)
    return g


def _find_shortest_path_nodes(active_graph: Dict[str, Dict], source_candidates: List[str], target_candidates: List[str]) -> Set[str]:
    if not HAS_NETWORKX:
        return set()
    graph = _build_nx_graph(active_graph)
    if graph is None:
        return set()

    best_path = None
    for src in source_candidates[:5]:
        for dst in target_candidates[:5]:
            if src not in graph or dst not in graph or src == dst:
                continue
            try:
                path = nx.shortest_path(graph, source=src, target=dst)
                if best_path is None or len(path) < len(best_path):
                    best_path = path
            except Exception:
                continue

    return set(best_path or [])


async def selector_agent_enhanced(
    target_repo: str,
    technical_query: str,
    multi_graph: Dict[str, Dict],
    vector_stores: Dict[str, VectorEmbeddingStore],
    query_type: str = "FUNCTIONAL_AREA",
    hints: List[str] = None,
    summarizer: ProjectSummarizer = None
) -> List[str]:
    """Enhanced selector with DYNAMIC SCALING based on graph size."""
    global LAST_SELECTOR_NODES
    print(f"🗂️ Selector: Strategy [{query_type}] in [{target_repo}]...", flush=True)
    
    # --- 1. Graph Preparation (Keep existing logic) ---
    if target_repo == "ALL" or target_repo not in multi_graph:
        active_graph = {}
        active_stores = {}
        local_to_global = {}
        for r_name, g_data in multi_graph.items():
            for node_id, node_data in g_data.items():
                gid = node_data.get('global_id', _global_node_id(r_name, node_data.get('file', ''), node_data.get('symbol', {}).get('name', node_id.split('::')[-1])))
                local_to_global[(r_name, node_id)] = gid

        for r_name, g_data in multi_graph.items():
            for node_id, node_data in g_data.items():
                gid = local_to_global[(r_name, node_id)]
                normalized_node = dict(node_data)
                normalized_node['dependencies'] = [local_to_global[(r_name, d)] for d in node_data.get('dependencies', []) if (r_name, d) in local_to_global]
                normalized_node['callers'] = [local_to_global[(r_name, c)] for c in node_data.get('callers', []) if (r_name, c) in local_to_global]
                active_graph[gid] = normalized_node
            active_stores[r_name] = vector_stores.get(r_name)
        if '__global__' in vector_stores:
            active_stores['__global__'] = vector_stores['__global__']
    else:
        active_graph = {k: v for k, v in multi_graph[target_repo].items()}
        active_stores = {target_repo: vector_stores.get(target_repo)}
    
    if not active_graph and query_type != "HIGH_LEVEL":
        return []

    # --- 2. Calculate Dynamic Limits ---
    # Filter for code nodes only (ignore folders/external libs for counting)
    code_nodes_only = [n for n in active_graph.values() if n.get('type') in ('function', 'class', 'method')]
    total_code_nodes = len(code_nodes_only)

    # SCALING LOGIC: 5% of graph, min 5, max 30
    dynamic_limit = max(5, min(30, int(total_code_nodes * 0.05)))
    
    # SCALING LOGIC FOR VECTORS: Increase K for larger graphs
    dynamic_k = max(TOP_K_SEEDS, min(15, int(total_code_nodes * 0.02)))

    print(f"    📊 Graph Size: {total_code_nodes} code nodes. Dynamic Limit: {dynamic_limit}. Vector K: {dynamic_k}")

    # --- 3. Memory Match (Keep existing) ---
    memory_match = _find_memory_match(technical_query)
    if memory_match and active_graph:
        mem_nodes = set(n for n in memory_match.get('selected_nodes', []) if n in active_graph)
        if mem_nodes:
            print("    🧠 Reusing memory-augmented retrieval hints")
            LAST_SELECTOR_NODES = list(mem_nodes)
            return build_context_with_budget(mem_nodes, active_graph, target_repo)

    # --- 4. Impact Analysis (Keep existing) ---
    if query_type == "IMPACT" and active_graph:
        specific_seeds = _extract_specific_targets(technical_query, hints or [], active_graph)
        if specific_seeds:
            LAST_SELECTOR_NODES = list(specific_seeds)
            return _get_blast_radius(next(iter(specific_seeds)), active_graph, max_depth=3)

    selected_nodes = set()
    extra_context = []
    explanation_weight_mode = query_type == "HIGH_LEVEL" or bool(re.search(r"explain|how\s+.*works|architecture", technical_query, re.IGNORECASE))
    
    # --- STRATEGY 0: DIRECT SYMBOL LOOKUP (Corrected) ---
    if active_graph:
        direct_hits = _extract_specific_targets(technical_query, hints or [], active_graph)
        if direct_hits:
            print(f"    🎯 Direct Symbol Hit: Found {len(direct_hits)} matches.")
            # If specifically asked for these, return immediately
            if query_type == "SPECIFIC":
                 return build_context_with_budget(
                    _priority_graph_traverse(direct_hits, active_graph, technical_query, max_depth=1), 
                    active_graph, 
                    target_repo
                )
            # Otherwise, add to selection and continue
            seed_nodes = _priority_graph_traverse(direct_hits, active_graph, technical_query, max_depth=1)
            selected_nodes.update(seed_nodes)

    # ... inside selector_agent_enhanced ...
    if query_type == "IMPACT" or "remove" in technical_query.lower():
        anchor_nodes = _extract_specific_targets(technical_query, hints, active_graph)
        if anchor_nodes:
            # 1. Provide the Tree Map
            extra_context.append(_build_impact_tree_text(anchor_nodes, active_graph))
            # 2. Grab the actual code for the anchors and their immediate callers
            selected_nodes.update(anchor_nodes)
            selected_nodes.update(_priority_graph_traverse(anchor_nodes, active_graph, technical_query, max_depth=2, direction="callers"))
    
    # --- Strategy 1: HIGH_LEVEL Summaries ---
    if query_type == "SPECIFIC" and summarizer:
        extra_context.append(f"=== PROJECT ARCHITECTURAL OVERVIEW ===\n\n{summarizer.project_summary}")
        file_sum_str = "\n".join([f"- {f}: {s}" for f, s in list(summarizer.file_summaries.items())[:20]])
        folder_sum_str = "\n".join([f"- {f}: {s}" for f, s in list(summarizer.folder_summaries.items())[:20]])
        extra_context.append(f"=== FILE SUMMARIES ===\n\n{file_sum_str}")
        if folder_sum_str:
            extra_context.append(f"=== FOLDER SUMMARIES ===\n\n{folder_sum_str}")

    # --- STRATEGY 1.5: CENTRALITY SWEEP (Dynamic) ---
    # Trigger: "High Level" OR "List..." OR "What are useful functions"
    if query_type == "HIGH_LEVEL" or (not hints and any(k in technical_query.lower() for k in ["list", "useful", "main", "core"])):
        print(f"    🌟 Performing Centrality Sweep (Top {dynamic_limit} nodes)...")
        
        # Sort by (Centrality * 0.7 + Documentation Quality * 0.3)
        # We want important nodes that are ALSO well-documented
        code_nodes_only.sort(
            key=lambda n: (float(n.get('centrality_score', 0)) * 0.7 + float(n.get('comment_quality', 0)) * 0.3), 
            reverse=True
        )
        
        # Select using DYNAMIC LIMIT
        top_central_nodes = [n.get('global_id') for n in code_nodes_only[:dynamic_limit]]
        selected_nodes.update(top_central_nodes)

    # --- Strategy 2: SPECIFIC (Backward Traversal) ---
    if (query_type == "SPECIFIC" or hints) and active_graph:
        specific_seeds = _extract_specific_targets(technical_query, hints or [], active_graph)
        if specific_seeds:
            selected_nodes.update(_priority_graph_traverse(
                specific_seeds,
                active_graph,
                technical_query,
                max_depth=2 if query_type in ["SPECIFIC", "HIGH_LEVEL"] else BACKWARD_TRAVERSAL_DEPTH,
                direction="dependencies",
                explanation_weight_mode=explanation_weight_mode
            ))
            if re.search(r"who\s+uses|callers?|used\s+by|impact", technical_query, re.IGNORECASE):
                selected_nodes.update(_priority_graph_traverse(
                    specific_seeds,
                    active_graph,
                    technical_query,
                    max_depth=3,
                    direction="callers",
                    explanation_weight_mode=explanation_weight_mode
                ))

    # Strategy 3: Vector Fallback / FUNCTIONAL_AREA
    if not selected_nodes and active_graph:
        seed_nodes = set()
        seed_scores = {}
        for repo, store in active_stores.items():
            if store and HAS_FAISS:
                # Use DYNAMIC_K here
                search_results = await store.search(technical_query, k=dynamic_k)
                for result_id in search_results:
                    full_id = result_id # (Simplify for brevity, assume ID mapping logic from original is here)
                    if repo != '__global__' and target_repo == "ALL":
                         node = multi_graph.get(repo, {}).get(result_id)
                         if node: full_id = node.get('global_id', result_id)
                    
                    if full_id in active_graph:
                        seed_nodes.add(full_id)
                        if store.last_scores.get(result_id) is not None:
                            seed_scores[full_id] = max(seed_scores.get(full_id, 0.0), store.last_scores[result_id])
        
        if not seed_nodes:
            seed_nodes = await llm_seed_selection(technical_query, active_graph)
            
        depth = 2 if explanation_weight_mode else (3 if query_type == "FUNCTIONAL_AREA" else MAX_RECURSION_DEPTH)
        selected_nodes.update(_priority_graph_traverse(
            seed_nodes,
            active_graph,
            technical_query,
            max_depth=depth,
            direction="dependencies",
            seed_scores=seed_scores,
            explanation_weight_mode=explanation_weight_mode
        ))

    # Strategy 4: Multi-hop path retrieval for endpoint->db style queries.
    if active_graph and re.search(r"endpoint|route|api", technical_query, re.IGNORECASE) and re.search(r"db|database|sql|repository", technical_query, re.IGNORECASE):
        endpoint_candidates = []
        db_candidates = []
        for nid, node in active_graph.items():
            blob = f"{nid} {node.get('file', '')} {node.get('code', '')[:200]}".lower()
            if any(tok in blob for tok in ["route", "controller", "endpoint", "@app.", "router."]):
                endpoint_candidates.append(nid)
            if any(tok in blob for tok in ["db", "database", "sql", "repository", "query", "insert", "update"]):
                db_candidates.append(nid)

        path_nodes = _find_shortest_path_nodes(active_graph, endpoint_candidates, db_candidates)
        selected_nodes.update(path_nodes)

    LAST_SELECTOR_NODES = list(selected_nodes)
    context_strings = build_context_with_budget(selected_nodes, active_graph, target_repo)
    return extra_context + context_strings

async def llm_seed_selection(query: str, active_graph: Dict) -> Set[str]:
    """Fallback LLM-based seed selection."""
    keys = list(active_graph.keys())[:800]
    
    menu = []
    for node_id in keys:
        data = active_graph[node_id]
        preview = (data.get('docstring') or data.get('comments_above') or '').strip().replace('\n', ' ')
        preview = preview[:120] if preview else 'No-comment-preview'
        menu.append(f"ID: {node_id} | Type: {data['type']} | Comment: {preview}")
    
    menu_str = "\n".join(menu)
    
    prompt = f"""
You are a Code Navigator.
Query: "{query}"

Available Nodes:
{menu_str}

TASK: Select 3-10 STARTING NODES (IDs) most relevant to the query.
Return JSON: {{"seed_nodes": ["node_id_1", "node_id_2"]}}
"""
    
    try:
        res = await safe_chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        seed_nodes = json.loads(res.choices[0].message.content).get('seed_nodes', [])
        return set(seed_nodes)
    except:
        return set()

def build_context_with_budget(
    selected_nodes: Set[str],
    active_graph: Dict,
    target_repo: str
) -> List[str]:
    """Build context strings with token budgeting using dynamic code slicing."""

    files_context = defaultdict(lambda: {"globals": "", "nodes": []})

    # Pull immediate dependencies as slice support nodes (program slicing).
    expanded_nodes = set(selected_nodes)
    for node_id in list(selected_nodes):
        node = active_graph.get(node_id)
        if not node:
            continue
        for dep_id in node.get('dependencies', [])[:5]:
            if dep_id in active_graph:
                expanded_nodes.add(dep_id)

    for node_id in expanded_nodes:
        if node_id not in active_graph:
            continue

        node = active_graph[node_id]
        repo_name = node.get('repo', '')
        rel_file_path = node.get('file', '')

        if target_repo == "ALL":
            display_name = f"[{repo_name}] {rel_file_path}"
        else:
            display_name = rel_file_path

        files_context[display_name]["globals"] = node.get('globals', '')
        files_context[display_name]["nodes"].append(node)

    context_blocks = []
    for fname, data in files_context.items():
        block = f"############################################################\n"
        block += f"### FILE: {fname}\n"
        block += f"### Sliced symbols in this block: {len(data['nodes'])}\n"
        block += f"############################################################\n\n"

        block += reconstruct_file_slice(data['globals'], data['nodes'])
        block += "\n\n"
        context_blocks.append(block)

    budgeted_context = truncate_to_token_budget(context_blocks, MAX_CONTEXT_TOKENS)
    return budgeted_context


async def llm_assess_context(user_query: str, context_strings: List[str]) -> Dict[str, Any]:
    """Assess whether current retrieved context is sufficient to answer."""
    preview = "\n\n".join(context_strings[:4])[:6000]
    prompt = f"""
You are a retrieval planner for a code reasoning system.
User query: {user_query}
Current context preview:
{preview}

Return strict JSON with:
{{
  "status": "ANSWERABLE" | "NEED_MORE_CONTEXT",
  "missing_info_query": "targeted follow-up query if more context is needed",
  "reason": "short reason"
}}
"""
    try:
        res = await safe_chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"status": "NEED_MORE_CONTEXT", "missing_info_query": user_query, "reason": "fallback"}


async def autonomous_answering_loop(
    user_query: str,
    reframer_res: Dict[str, Any],
    multi_graph: Dict[str, Dict],
    vector_stores: Dict[str, VectorEmbeddingStore],
    summarizer: ProjectSummarizer,
    max_turns: int = MAX_REASONING_TURNS
) -> str:
    """Agentic RAG loop: Plan -> Retrieve -> Evaluate -> Loop -> Answer."""
    context_pool = []
    seen = set()
    active_query = reframer_res.get("query", user_query)

    for turn in range(max_turns):
        print(f"🔁 Reasoning turn {turn + 1}/{max_turns}...")

        chunks = await selector_agent_enhanced(
            reframer_res.get("target_repo", "ALL"),
            active_query,
            multi_graph,
            vector_stores,
            query_type=reframer_res.get("query_type", "FUNCTIONAL_AREA"),
            hints=reframer_res.get("hints", []),
            summarizer=summarizer
        )

        for chunk in chunks:
            digest = hashlib.md5(chunk.encode('utf-8', errors='ignore')).hexdigest()
            if digest not in seen:
                seen.add(digest)
                context_pool.append(chunk)

        if not context_pool:
            continue

        assessment = await llm_assess_context(user_query, context_pool)
        if assessment.get("status") == "ANSWERABLE":
            answer = await answering_agent(user_query, context_pool)
            _record_reasoning_memory(user_query, LAST_SELECTOR_NODES, answer)
            return answer

        active_query = assessment.get("missing_info_query") or active_query

    if context_pool:
        answer = await answering_agent(user_query, context_pool)
        _record_reasoning_memory(user_query, LAST_SELECTOR_NODES, answer)
        return answer
    return "I tried multiple retrieval passes but could not find enough evidence to answer confidently."

def compress_context_strings(context_strings: List[str], token_budget: int = CONTEXT_COMPRESSION_TRIGGER_TOKENS) -> List[str]:
    """Compress context opportunistically by removing repetitive globals and oversize slices."""
    if not context_strings:
        return context_strings

    full = "\n".join(context_strings)
    if count_tokens(full) <= token_budget:
        return context_strings

    compressed = []
    seen_globals = set()
    for block in context_strings:
        b = block
        globals_match = re.search(r"--- FILE GLOBALS, IMPORTS AND CONSTANTS ---\n(.*?)\n=+", b, re.DOTALL)
        if globals_match:
            g = globals_match.group(1).strip()
            g_hash = hashlib.md5(g.encode('utf-8', errors='ignore')).hexdigest()
            if g_hash in seen_globals:
                b = b.replace(globals_match.group(0), "")
            else:
                seen_globals.add(g_hash)

        if len(b) > 10000:
            lines = b.splitlines()
            b = "\n".join(lines[:120] + ["... [compressed] ..."] + lines[-120:])

        compressed.append(b)

    return truncate_to_token_budget(compressed, token_budget)


# --- Reframer ---

async def reframer_agent(user_query: str, chat_history: List[Dict], available_repos: List[str]) -> Dict[str, Any]:
    """Detect target repo, query type, and architectural hints."""
    print("🧠 Reframer: Analyzing query...", flush=True)
    
    history_text = ""
    for turn in chat_history[-3:]:
        history_text += f"{turn['role'].upper()}: {turn['content']}\n"
    
    repo_list_str = ", ".join(available_repos)
    
    prompt = f"""
You are a Technical Assistant managing multiple repositories.
Available Repositories: [{repo_list_str}]

Conversation History:
{history_text}

Current Query: "{user_query}"

TASK:
1. Determine the TARGET_REPO (name or 'ALL').
2. Classify QUERY_TYPE:
   - 'SPECIFIC': User asks about a specific function, class, or bug (e.g., "how does X work?").
   - 'FUNCTIONAL_AREA': User asks about a feature/module (e.g., "auth system", "database layer").
   - 'HIGH_LEVEL': User asks for lists, overviews, or "how things work" generally (e.g., "list useful functions", "overview of repo").
3. Extract HINTS: 
   - IF QUERY_TYPE is 'HIGH_LEVEL', return []. Do NOT carry over previous hints.
   - Otherwise, list filenames/symbols mentioned.
4. Rewrite the QUERY for better retrieval.

OUTPUT FORMAT:
TARGET_REPO: <repo>
QUERY_TYPE: <type>
HINTS: ["hint1", "hint2"] or []
QUERY: <rewritten_query>
"""
    
    res = await safe_chat_completion(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    content = res.choices[0].message.content.strip()
    
    result = {
        "target_repo": "ALL",
        "query_type": "FUNCTIONAL_AREA",
        "hints": [],
        "query": user_query
    }
    
    match_repo = re.search(r"TARGET_REPO:\s*(.+)", content)
    if match_repo:
        result["target_repo"] = match_repo.group(1).strip()
    
    match_type = re.search(r"QUERY_TYPE:\s*(.+)", content)
    if match_type:
        result["query_type"] = match_type.group(1).strip()
        
    match_hints = re.search(r"HINTS:\s*(\[.+\])", content)
    if match_hints:
        try:
            result["hints"] = json.loads(match_hints.group(1))
        except:
            pass
    
    match_query = re.search(r"QUERY:\s*(.+)", content, re.DOTALL)
    if match_query:
        result["query"] = match_query.group(1).strip()
    
    print(f"   ↳ Target: [{result['target_repo']}] | Type: {result['query_type']} | Hints: {result['hints']}")
    return result

# --- Answering Agent ---

async def answering_agent(user_query: str, context_strings: List[str]) -> str:
    """Generate answer with streaming output."""
    print("📝 Answering Agent: Generating response...", flush=True)
    
    context_strings = compress_context_strings(context_strings)
    full_context = "\n".join(context_strings)
    context_tokens = count_tokens(full_context)
    print(f"   📊 Context size: {context_tokens} tokens")
    
    messages = [
        {"role": "system", "content": "You are a senior developer. Answer based strictly on the provided Code Context. Be specific and cite file names when relevant."},
        {"role": "user", "content": f"Query: {user_query}\n\nCode Context:\n{full_context}"}
    ]
    
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        print("\n" + "="*60)
        print("✅ ANSWER:")
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print("\n" + "="*60)
        return full_response
        
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        res = await safe_chat_completion(model=MODEL_NAME, messages=messages)
        return res.choices[0].message.content

# --- Cache Management ---

def save_cache(multi_graph: Dict, vector_stores: Dict, repo_hashes: Dict, summarizer: ProjectSummarizer):
    """Save graph, embeddings, and summaries to cache."""
    print("💾 Saving cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_data = {
        'timestamp': time.time(),
        'repo_hashes': repo_hashes,
        'graph': multi_graph,
        'file_hashes': repo_hashes.get('file_hashes', {})
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)

    # Save vector indices
    for repo_name, store in vector_stores.items():
        index_path = os.path.join(CACHE_DIR, f"{repo_name}.faiss")
        store.save(index_path)
    
    # Save summaries
    summarizer.save(os.path.join(CACHE_DIR, "summaries.json"))

    print(f"   ✅ Cache saved to {CACHE_DIR}")

def load_cache(current_hashes: Dict, summarizer: ProjectSummarizer) -> Optional[Tuple[Dict, Dict, Dict, bool]]:
    """Load cache if valid."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        cached_hashes = cache_data.get('repo_hashes', {})
        hash_match = cached_hashes == current_hashes
        if not hash_match:
            print("⚠️ Repository changes detected, attempting partial re-index")

        print("✅ Loading from cache...")
        multi_graph = cache_data['graph']
        cached_file_hashes = cache_data.get('file_hashes', {})
        
        vector_stores = {}
        index_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.faiss')]
        for index_file in index_files:
            store_key = index_file[:-6]
            store = VectorEmbeddingStore()
            index_path = os.path.join(CACHE_DIR, index_file)
            if os.path.exists(index_path):
                store.load(index_path)
                vector_stores[store_key] = store
        
        # Load summaries
        summarizer.load(os.path.join(CACHE_DIR, "summaries.json"))
        
        return multi_graph, vector_stores, cached_file_hashes, hash_match
    except Exception as e:
        print(f"⚠️ Cache load error: {e}")
        return None

# --- Main ---

async def main():
    chat_history = []
    init_cache_dir()
    summarizer = ProjectSummarizer()

    try:
        print("🔗 === GITHUB SOURCE CONFIGURATION === 🔗")
        gh_input = input("\n🐙 Enter GitHub Repos (comma-separated): ")
        if not gh_input.strip():
            return
        
        multi_repo_data, repo_hashes = await ingest_sources(gh_input)
        if not multi_repo_data:
            return

        current_file_hashes = compute_multi_repo_file_hashes(multi_repo_data)
        cache_key_payload = {"repo_hashes": repo_hashes, "file_hashes": current_file_hashes}

        cached = load_cache(cache_key_payload, summarizer)

        if cached:
            cached_graph, cached_vectors, cached_file_hashes, hash_match = cached
            if hash_match:
                multi_graph, vector_stores = cached_graph, cached_vectors
            else:
                multi_graph, vector_stores = await build_multi_symbol_graph(
                    multi_repo_data,
                    summarizer,
                    previous_graph=cached_graph,
                    previous_vector_stores=cached_vectors,
                    previous_file_hashes=cached_file_hashes,
                    current_file_hashes=current_file_hashes
                )
                save_cache(multi_graph, vector_stores, cache_key_payload, summarizer)
        else:
            multi_graph, vector_stores = await build_multi_symbol_graph(
                multi_repo_data,
                summarizer,
                current_file_hashes=current_file_hashes
            )
            save_cache(multi_graph, vector_stores, cache_key_payload, summarizer)
        
        available_repos = list(multi_graph.keys())
        
        # Check if we have any nodes
        total_nodes = sum(len(graph) for graph in multi_graph.values())
        
        if total_nodes == 0:
            print("\n⚠️ Warning: No code symbols were extracted from the repository.")
            print("\n   You can still ask questions, but responses may be limited.")
        
        print(f"\n✅ System ready! Found {total_nodes} symbols across {len(available_repos)} repo(s).")
        print("   Ask questions about your code.")
        
        while True:
            query = input("\n" + "-"*40 + "\n(Type 'exit' to quit) Question: ")
            if query.lower() in ['exit', 'quit']:
                break
            
            reframer_res = await reframer_agent(query, chat_history, available_repos)
            
            answer = await autonomous_answering_loop(
                query,
                reframer_res,
                multi_graph,
                vector_stores,
                summarizer,
                max_turns=MAX_REASONING_TURNS
            )
            
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        perform_cleanup()

if __name__ == "__main__":
    asyncio.run(main())
