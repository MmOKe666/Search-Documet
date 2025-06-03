import weaviate
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Загружаем модель и токенайзер один раз
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def query_weaviate(query_text):
    # Векторизуем запрос
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        vector = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0]

    # Подключаемся к Weaviate
    client = weaviate.Client("http://localhost:8080")

    # Выполняем семантический поиск по вектору
    result = client.query.get(
        "Document", ["metadata"]
    ).with_near_vector({
        "vector": vector.tolist()
    }).with_limit(5).do()

    return result

if __name__ == "__main__":
    query_text = input("Введите запрос: ")
    results = query_weaviate(query_text)

    print("\nРезультаты поиска:")
    documents = results["data"]["Get"]["Document"]
    for doc in documents:
        metadata = json.loads(doc["metadata"])
        print("    Компонент:", metadata.get("component"))
        print("    Название:", metadata.get("title"))
        print("    Дата:", metadata.get("date"))
        print()