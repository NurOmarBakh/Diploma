import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Инициализация модели для получения эмбеддингов.
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def load_texts_from_directory(directory):
    texts = []
    # Обработка файлов в отсортированном порядке.
    for fname in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, fname)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Разбиваем текст на куски по двойному переводу строки
                chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
                texts.extend(chunks)
        except Exception as e:
            print(f"Ошибка при обработке файла {fname}: {e}")
    return texts

def create_embeddings(texts):
    # Генерация эмбеддингов с индикатором прогресса (если поддерживается)
    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    return index

def save_index_and_texts(index, texts, index_path="vector_db/index.faiss", texts_path="vector_db/embeddings.pkl"):
    # Создаем директории, если их нет.
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(texts_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)

def main():
    kb_dir = "data/knowledge_base"
    texts = load_texts_from_directory(kb_dir)
    print(f"Загружено {len(texts)} кусочков текста из {kb_dir}.")
    embeddings = create_embeddings(texts)
    index = create_faiss_index(embeddings)
    save_index_and_texts(index, texts)
    print("[+] Индексация завершена")

if __name__ == "__main__":
    main()
