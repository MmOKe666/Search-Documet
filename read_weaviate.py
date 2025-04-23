import weaviate

# Подключение к локальному Weaviate
client = weaviate.Client("http://localhost:8080")

# Проверка подключения
if client.is_ready():
    print("✅ Подключено к Weaviate")
else:
    print("❌ Не удалось подключиться к Weaviate")

# Название класса, который ты создал (замени, если нужно)
class_name = "Document"

# Запрос всех объектов этого класса
result = client.query.get(class_name, ["content", "metadata"]).do()

# Вывод
import json
print(json.dumps(result, indent=2, ensure_ascii=False))