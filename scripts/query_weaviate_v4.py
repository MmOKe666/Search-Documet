import weaviate
from weaviate.classes.query import NearText

# Connect to Weaviate
client = weaviate.connect_to_local()  # Connect with default parameters

def query_weaviate(query_text):
    # Get the Document collection
    collection = client.collections.get("Document")
    
    # Create a query with near_text
    response = collection.query.near_text(
        query=query_text,  # In v4, you can directly pass the query text
        limit=5,
        return_properties=["title"]  # Specify which properties to return
    )
    
    # Get the objects from the response
    results = response.objects
    
    return results


if __name__ == "__main__":
    query_text = input("Введите запрос: ")
    results = query_weaviate(query_text)
    print("Результаты поиска:")
    for result in results:
        print(f"Title: {result.properties.get('title')}")
        print(f"Distance: {result.metadata.distance}")
        print("---")