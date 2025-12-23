from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
import json
import os

# Пути к папкам
json_folder = "C:/Savelii Volkovich/python/prog/Files/json"
json_path = os.path.join(json_folder, "documentation_mapping.json")

# Подключение к Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

# Загрузка модели для векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Создание класса, если его нет
class_obj = {
    "class": "Document",
    "description": "Документы с контентом и метаданными",
    "properties": [
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Основной текст документа"
        },
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Название документа"
        },
        {
            "name": "metadata",
            "dataType": ["text"],
            "description": "Метаданные документа в JSON"
        }
    ]
}

try:
    client.collections.get("Document")
    print("ℹ️ Класс Document уже существует")
except weaviate.exceptions.WeaviateQueryError:
    client.collections.create_from_dict(class_obj)
    print("✅ Класс Document создан")

def embed_and_store(text, metadata):
    # Векторизация текста
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0]

    # Загрузка в Weaviate
    documents = client.collections.get("Document")
    documents.data.insert(
        properties={
            "content": text,
            "title": metadata.get("title"),
            "metadata": json.dumps(metadata, ensure_ascii=False)
        },
        vector=embeddings.tolist()
    )


# Читаем JSON и загружаем документы
with open(json_path, encoding='utf-8') as json_file:
    documents = json.load(json_file)
    for doc in documents:
        text = doc["content"]
        metadata = doc["metadata"]
        embed_and_store(text, metadata)

print("✅ Все документы загружены в Weaviate.")