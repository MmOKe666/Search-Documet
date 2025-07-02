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

# Загрузка модели для векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Подключение к локальному Qdrant
client = QdrantClient(host="localhost", port=6333)

# Название коллекции
collection_name = "Document"

# Функция векторизации и сохранения
def embed_and_store(text, metadata):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]

    payload = {
        "content": text,
        "title": metadata.get("title"),
        "metadata": json.dumps(metadata, ensure_ascii=False)
    }

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings.tolist(),
        payload=payload
    )

    client.upsert(collection_name=collection_name, points=[point])

# Создание коллекции, если она не существует
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Коллекция Document создана")
else:
    print("ℹ️ Коллекция Document уже существует")

# Загрузка документов из JSON и векторизация
try:
    with open(json_path, encoding='utf-8') as json_file:
        documents = json.load(json_file)
        total = len(documents)
        for i, doc in enumerate(documents):
            try:
                text = doc["content"]
                metadata = doc["metadata"]
                embed_and_store(text, metadata)
                print(f"Processed {i + 1}/{total} documents")
            except Exception as e:
                print(f"Error processing document {i}: {e}")
except Exception as e:
    print(f"Error loading JSON file: {e}")

print("✅ Все документы загружены в Qdrant.")