"""
Semantic Search over Memory Files
Builds an embedding index over all markdown files in clawd/ and memory/
Supports natural language queries that understand meaning, not just keywords.

Usage:
  python semantic_search.py index          # Build/rebuild the index
  python semantic_search.py search "query" # Search with natural language
  python semantic_search.py search "query" --top 10  # More results
"""
import os, sys, json, hashlib, time
import numpy as np
from pathlib import Path

CLAWD_DIR = Path(os.path.expanduser("~/clawd"))
INDEX_PATH = CLAWD_DIR / "tools" / "semantic_index.npz"
META_PATH = CLAWD_DIR / "tools" / "semantic_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dims

# Files to index
SCAN_PATTERNS = [
    "*.md",
    "memory/*.md",
    "memory/**/*.md",
    "pipeline/*.py",
    "tools/*.py",
]
SKIP_FILES = {"node_modules", ".git", "__pycache__", "semantic_index.npz"}

def find_files():
    """Find all indexable files."""
    files = set()
    for pattern in SCAN_PATTERNS:
        for f in CLAWD_DIR.glob(pattern):
            if any(skip in str(f) for skip in SKIP_FILES):
                continue
            if f.is_file() and f.stat().st_size > 0:
                files.add(f)
    return sorted(files)

def chunk_file(filepath, chunk_size=500, overlap=100):
    """Split a file into overlapping chunks for better retrieval."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except:
        return []
    
    chunks = []
    lines = text.split("\n")
    current = []
    current_len = 0
    
    for i, line in enumerate(lines):
        current.append(line)
        current_len += len(line) + 1
        
        if current_len >= chunk_size:
            chunk_text = "\n".join(current)
            chunks.append({
                "file": str(filepath.relative_to(CLAWD_DIR)),
                "text": chunk_text,
                "start_line": i - len(current) + 2,
                "end_line": i + 1,
            })
            # Keep overlap
            overlap_lines = []
            ol = 0
            for l in reversed(current):
                overlap_lines.insert(0, l)
                ol += len(l) + 1
                if ol >= overlap:
                    break
            current = overlap_lines
            current_len = ol
    
    # Last chunk
    if current:
        chunk_text = "\n".join(current)
        if len(chunk_text.strip()) > 20:
            chunks.append({
                "file": str(filepath.relative_to(CLAWD_DIR)),
                "text": chunk_text,
                "start_line": len(lines) - len(current) + 1,
                "end_line": len(lines),
            })
    
    return chunks

def build_index():
    """Build the embedding index."""
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    files = find_files()
    print(f"Found {len(files)} files to index")
    
    all_chunks = []
    for f in files:
        chunks = chunk_file(f)
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    if not all_chunks:
        print("No chunks to index!")
        return
    
    texts = [c["text"] for c in all_chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    # Save index
    np.savez_compressed(str(INDEX_PATH), embeddings=embeddings)
    
    # Save metadata (without the full text, just file + lines)
    meta = []
    for c in all_chunks:
        meta.append({
            "file": c["file"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "preview": c["text"][:200],
        })
    
    # Also save file hashes for incremental updates
    file_hashes = {}
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        file_hashes[str(f.relative_to(CLAWD_DIR))] = h
    
    with open(META_PATH, "w") as f:
        json.dump({"meta": meta, "file_hashes": file_hashes, "model": MODEL_NAME,
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_chunks": len(all_chunks)}, f, indent=2)
    
    print(f"Index saved: {len(all_chunks)} chunks, {INDEX_PATH}")

def search(query, top_k=5):
    """Search the index with a natural language query."""
    from sentence_transformers import SentenceTransformer
    
    if not INDEX_PATH.exists() or not META_PATH.exists():
        print("No index found. Run: python semantic_search.py index")
        return []
    
    model = SentenceTransformer(MODEL_NAME)
    data = np.load(str(INDEX_PATH))
    embeddings = data["embeddings"]
    
    with open(META_PATH) as f:
        meta_data = json.load(f)
    meta = meta_data["meta"]
    
    q_emb = model.encode([query])
    
    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    scores = (normed @ q_norm.T).flatten()
    
    top_idx = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_idx:
        r = meta[idx].copy()
        r["score"] = float(scores[idx])
        results.append(r)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python semantic_search.py [index|search] [query]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "index":
        build_index()
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python semantic_search.py search 'your query'")
            sys.exit(1)
        query = sys.argv[2]
        top_k = 5
        if "--top" in sys.argv:
            top_k = int(sys.argv[sys.argv.index("--top") + 1])
        
        results = search(query, top_k)
        print(f"\nResults for: '{query}'\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['file']}:{r['start_line']}-{r['end_line']}")
            print(f"     {r['preview'][:120]}...")
            print()
    else:
        print(f"Unknown command: {cmd}")
