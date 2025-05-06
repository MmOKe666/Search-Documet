from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import json
import os

# Пути к папкам
json_folder = os.path.normpath("../Files/json")
json_path = os.path.join(json_folder, "documentation_mapping.json")

# Загрузка модели для векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Replace with a model better suited for Russian
# tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# Connect to Weaviate
client = weaviate.connect_to_local()  # Connect with default parameters

# Define the function before it's called
def embed_and_store(text, metadata):
    # Tokenize with a higher max length for Russian text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                      padding=True, max_length=512)  # Increased max_length

    with torch.no_grad():
        # Get embeddings from the last hidden state
        outputs = model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]

    # Загрузка в Weaviate
    docs = client.collections.get("Document")
    docs.data.insert(
        properties={
            "content": text,
            "title": metadata.get("title"),
            "metadata": json.dumps(metadata, ensure_ascii=False)  # ensure_ascii=False is good for Russian
        },
        vector=embeddings.tolist()
    )


try:
    # Check if collection exists and create if needed
    if not client.collections.exists("Document"):
        # Create collection with properties
        client.collections.create(
            "Document",
            description="Документы с контентом и метаданными",
            properties=[
                Property(name="content", data_type=DataType.TEXT, description="Основной текст документа"),
                Property(name="title", data_type=DataType.TEXT, description="Название документа"),
                Property(name="metadata", data_type=DataType.TEXT, description="Метаданные документа в JSON")
            ]
        )
        print("✅ Класс Document создан")
    else:
        print("ℹ️ Класс Document уже существует")
    
    # Читаем JSON и загружаем документы
    try:
        with open(json_path, encoding='utf-8') as json_file:
            documents = json.load(json_file)
            total = len(documents)
            for i, doc in enumerate(documents):
                try:
                    text = doc["content"]
                    metadata = doc["metadata"]
                    embed_and_store(text, metadata)
                    print(f"Processed {i+1}/{total} documents")
                except Exception as e:
                    print(f"Error processing document {i}: {e}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")

finally:
    client.close()  # Ensure the connection is closed

print("✅ Все документы загружены в Weaviate.")