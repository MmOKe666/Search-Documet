import bs4
from langchain_community.document_loaders import WebBaseLoader
import weaviate
from weaviate.classes.query import MetadataQuery  # Import MetadataQuery directly
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.config import Property, DataType
import json
import os
import torch



b4_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_paths=("https://ip-office.com",),
    bs_kwargs={"parse_only": b4_strainer},
    encoding="utf-8"
)

docs = loader.load()

print(f"Total charactes: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(docs)

print(f"Total splits: {len(all_splits)} ")

# path to folder with  pdf files
pdf_folder = os.path.normpath("../Files/pdf")

# Загрузка модели для векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Replace with a model better suited for Russian
# tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")


# Connect to Weaviate
client = weaviate.connect_to_local()

print(client.is_ready())  # Should print: `True`


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
        with open(pdf_folder, encoding='utf-8') as pdf_file:
            documents = json.load(pdf_file)
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