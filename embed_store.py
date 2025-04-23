from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
import json
import os

# Пути к папкам
json_folder = "C:/Савелий Волкович/python/prog/Files/json"
json_path = os.path.join(json_folder, "documentation_mapping.json")

# Подключение к Weaviate
client = weaviate.Client("http://localhost:8080")

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

existing_schema = client.schema.get()
if not any(c["class"] == "Document" for c in existing_schema.get("classes", [])):
    client.schema.create_class(class_obj)
    print("✅ Класс Document создан")
else:
    print("ℹ️ Класс Document уже существует")

def embed_and_store(text, metadata):
    # Векторизация текста
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0]

    # Загрузка в Weaviate
    client.data_object.create(
        data_object={
            "content": text,
            "title": metadata.get("title"),
            "metadata": json.dumps(metadata, ensure_ascii=False)
        },
        class_name="Document",
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