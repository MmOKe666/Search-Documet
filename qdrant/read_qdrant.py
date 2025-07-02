from qdrant_client import QdrantClient
import json

# Подключение к локальному Qdrant
client = QdrantClient(host="localhost", port=6333)

# Проверка подключения — делаем безопасный запрос
try:
    collections = client.get_collections()
    print("✅ Подключено к Qdrant")
except Exception as e:
    print("❌ Не удалось подключиться к Qdrant:", e)
    exit()

# Название коллекции
collection_name = "Document"

# Получаем все документы с полями content и metadata
scroll_result, _ = client.scroll(
    collection_name=collection_name,
    limit=100,
    with_payload=True,  # Загружаем все поля payload
    with_vectors=False  # Если не нужны векторы
)

# Форматируем результат
output = []
for point in scroll_result:
    payload = point.payload or {}
    output.append({
        "content": payload.get("content"),
        "metadata": payload.get("metadata")
    })

# Печать в JSON-формате
print(json.dumps(output, indent=2, ensure_ascii=False))