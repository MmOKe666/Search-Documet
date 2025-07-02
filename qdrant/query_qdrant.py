from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Загрузка той же модели, что использовалась при векторизации
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Название коллекции
collection_name = "Document"

# Функция для векторизации запроса
def vectorize_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    return embeddings.tolist()

# Функция запроса к Qdrant
def query_qdrant(query_text):
    try:
        # Подключение к локальному Qdrant
        client = QdrantClient(host="localhost", port=6333)

        # Проверка существования коллекции
        if not client.collection_exists(collection_name=collection_name):
            print(f"Error: Коллекция '{collection_name}' не найдена в Qdrant.")
            return []

        # Векторизация запроса
        query_vector = vectorize_text(query_text)

        # Поиск ближайших векторов
        response = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=2,
            with_payload=True,
            with_vectors=False
        )

        # DEBUG: Печать информации о расстоянии
        if response:
            first = response[0]
            print("DEBUG: First result metadata values:")
            print(f"  Distance: {first.score:.4f}")

        return response

    except Exception as e:
        print(f"Ошибка при запросе к Qdrant: {e}")
        return []

# Основной блок
if __name__ == "__main__":
    query_text = input("Введите запрос: ")
    results = query_qdrant(query_text)

    if results:
        print("Результаты поиска:")

        exact_matches = []
        other_matches = []

        for result in results:
            payload = result.payload or {}
            metadata_str = payload.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
                component = metadata.get("component", "")
                if query_text.lower() == component.lower():
                    exact_matches.append(result)
                else:
                    other_matches.append(result)
            except:
                other_matches.append(result)

        sorted_results = exact_matches + other_matches

        for result in sorted_results:
            payload = result.payload or {}
            metadata_str = payload.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
                component = metadata.get("component", "")
                if query_text.lower() == component.lower():
                    print(f"Title: {payload.get('title', 'No title')} [EXACT MATCH]")
                else:
                    print(f"Title: {payload.get('title', 'No title')}")
                if 'component' in metadata:
                    print(f"Component: {metadata['component']}")
                if 'version' in metadata:
                    print(f"Version: {metadata['version']}")
                print(f"Distance: {result.score:.4f}")
            except:
                print(f"Title: {payload.get('title', 'No title')}")
            content = payload.get("content", "")
            if content:
                snippet = content[:200] + "..." if len(content) > 200 else content
                print(f"Content snippet: {snippet}")
            print("---")
    else:
        print("Нет результатов или произошла ошибка.")