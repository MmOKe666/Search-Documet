from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
from weaviate.classes.config import Property, DataType
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
client = weaviate.connect_to_local()

print(client.is_ready())  # Should print: `True`

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
            # properties= [
            #     {
            #         "name": "content", 
            #         "dataType": ["text"], 
            #         "description": "Основной текст документа"
            #     },
            #     {
            #         "name": "title", 
            #         "dataType": ["text"], 
            #         "description": "Название документа"
            #     },
            #     {
            #         "name": "metadata", 
            #         "dataType": ["text"], 
            #         "description": "Метаданные документа в JSON"
            #     }
            # ]
            # the below lines not yet work in python 3.13
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

def print_schema():
    """
    Print the schema of the Weaviate instance using v4 API
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        # Get all collections
        collections = client.collections.get_all()
        
        print("=== WEAVIATE SCHEMA ===")
        print(f"Found {len(collections)} collections:")
        
        # Iterate through each collection
        for collection in collections:
            print(f"\n📚 Collection: {collection.name}")
            print(f"   Description: {collection.description}")
            print(f"   Vector dimension: {collection.vector_indexing.vector_index_config.dimensions}")
            
            # Get properties
            properties = collection.properties
            print(f"   Properties ({len(properties)}):")
            
            for prop in properties:
                data_type = prop.data_type
                if isinstance(data_type, list):
                    data_type = ", ".join(data_type)
                print(f"   - {prop.name}: {data_type}")
                if prop.description:
                    print(f"     Description: {prop.description}")
            
            # Get additional configuration
            print(f"   Vector index type: {collection.vector_indexing.vector_index_type}")
            print(f"   Sharding strategy: {collection.sharding.strategy}")
            
            # Get count of objects
            try:
                count = collection.aggregate.over_all().objects_count
                print(f"   Total objects: {count}")
            except Exception as e:
                print(f"   Error getting object count: {e}")
    
    except Exception as e:
        print(f"Error retrieving schema: {e}")
    finally:
        client.close()

# Uncomment the line below to print the schema after loading documents
# print_schema()