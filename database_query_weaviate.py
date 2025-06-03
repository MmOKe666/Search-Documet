import weaviate


def query_weaviate(query_text):
    client = weaviate.Client("http://localhost:8080")
    schema = client.schema.get()
    print(schema)
    result = client.query.get(
        "Document", ["title"]
    ).with_near_text({
        "concepts": [query_text],
        "certainty": 0.7  # Optional: specify the certainty level
    }).with_limit(5).do()

    return result

if __name__ == "__main__":
     query_text = input("Enter your query: ")
     results = query_weaviate(query_text)
     print("Query Results:")
     print(results)