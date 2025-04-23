import weaviate


def query_weaviate(query_text):
    client = weaviate.Client("http://localhost:8080")
    result = client.query.get(
        "Document", ["title"]
    ).with_near_text({"concepts": [query_text]}).with_limit(5).do()

    return result


if __name__ == "__main__":
    query_text = input("Введите запрос: ")
    results = query_weaviate(query_text)
    print("Результаты поиска:")
    print(results)