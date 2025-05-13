import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "vector_db1"

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

try:
    with open(f"{VECTOR_DB_PATH}/embeddings.pkl", "rb") as f:
        texts = pickle.load(f)
    index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")
except Exception as e:
    print(f"Ошибка загрузки векторной базы: {e}")
    texts = []
    index = None

def get_relevant_chunks(query: str, k=10):
    if index is None or not texts:
        return []
    query_vec = EMBEDDING_MODEL.encode([query])
    scores, idxs = index.search(np.array(query_vec), k)
    return [texts[i] for i in idxs[0] if i < len(texts)]

