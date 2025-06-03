from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch
import json
import os
import uuid

# Пути к папкам
json_folder = os.path.normpath("../Files/json")
json_path = os.path.join(json_folder, "documentation_mapping.json")

# Подключение к Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

# Название коллекции
collection_name = "documents"

# Загрузка модели для векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Проверка и создание коллекции
if not qdrant.collection_exists(collection_name=collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("✅ Коллекция создана")
else:
    print("ℹ️ Коллекция уже существует")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0]
    return embeddings

# Загрузка документов из JSON и вставка в Qdrant
with open(json_path, encoding='utf-8') as json_file:
    documents = json.load(json_file)
    points = []
    for doc in documents:
        text = doc["content"]
        metadata = doc["metadata"]
        vector = embed_text(text)
        payload = {
            "content": text,
            "title": metadata.get("title"),
            "metadata": metadata
        }
        point = PointStruct(
            id=str(uuid.uuid4()),  # уникальный ID
            vector=vector,
            payload=payload
        )
        points.append(point)

    qdrant.upsert(collection_name=collection_name, points=points)

print("✅ Все документы загружены в Qdrant.")