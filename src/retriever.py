# src/retriever.py

import os
import pickle
import logging
import yaml

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_config():
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retriever")

cfg  = load_config()
ROOT = project_root()

IDX_DIR   = os.path.join(ROOT, cfg["data"]["faiss_index_dir"])
IDX_PATH  = os.path.join(IDX_DIR, "index.faiss")
META_PATH = os.path.join(IDX_DIR, "metadata.pkl")

TOP_K  = cfg["retrieve"]["top_k"]
RR_K   = cfg["retrieve"]["rerank_k"]
CE_MOD = cfg["retrieve"]["cross_encoder_model"]
EMB_MOD = cfg["embed"]["model_name"]

logger.info(f"Loading index from {IDX_PATH}")
index = faiss.read_index(IDX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)
logger.info(f"Loaded {len(metadata)} entries")

logger.info(f"Loading embedder {EMB_MOD}")
embedder = SentenceTransformer(EMB_MOD)
reranker = CrossEncoder(CE_MOD) if CE_MOD else None

def retrieve(query: str) -> list[dict]:
    qv = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(qv, TOP_K)
    D, I = D[0], I[0]

    docs = []
    for dist, idx in zip(D, I):
        entry = metadata[idx].copy()
        docs.append({
            "score": float(dist),
            "text":  entry.pop("text"),
            **entry
        })

    if reranker:
        pairs  = [[query, d["text"]] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = []
        for d, s in ranked[:RR_K]:
            d["score"] = float(s)
            docs.append(d)

    return docs

if __name__ == "__main__":
    q = input("Q: ")
    for i, d in enumerate(retrieve(q), 1):
        print(i, d["score"], d["page_url"], "chunk", d["chunk_id"])
        print(d["text"][:100], "...\n")
