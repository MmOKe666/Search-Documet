from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
import json

# === 1. Подключение к Qdrant ===
qdrant = QdrantClient(host="localhost", port=6333)
collection_name = "documents"

# === 2. Загрузка модели для векторизации ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    """Векторизация текста"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0]
    return embeddings

# === 3. Ввод вопроса ===
question = input(" Введите ваш вопрос: ")

# === 4. Векторизация запроса ===
question_vector = embed_text(question)

# === 5. Поиск по вектору ===
results = qdrant.search(
    collection_name=collection_name,
    query_vector=question_vector,
    limit=5,
    with_payload=True
)

# === 6. Вывод результатов ===
if results:
    print("\n Наиболее релевантные документы:")
    for i, result in enumerate(results, 1):
        print(f"\n Результат {i} (score: {result.score:.4f}):")
        print(f" Заголовок: {result.payload.get('title')}")
        print(f" Контент: {result.payload.get('content')[:300]}...")
else:
    print("❌ Ничего не найдено.")