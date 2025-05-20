import weaviate
from weaviate.classes.query import MetadataQuery  # Import MetadataQuery directly
from transformers import AutoTokenizer, AutoModel
import torch

# Load the same model used for embedding during storage
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def vectorize_text(text):
    # Tokenize text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                      padding=True, max_length=512)
    
    with torch.no_grad():
        # Get embeddings from the last hidden state
        outputs = model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    return embeddings.tolist()

def query_weaviate(query_text):
    try:
        # Connect to Weaviate - make sure Weaviate is running locally
        client = weaviate.connect_to_local()
        
        # Check if the collection exists
        if not client.collections.exists("Document"):
            print("Error: 'Document' collection does not exist in Weaviate")
            return []
        
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Vectorize the query text
        query_vector = vectorize_text(query_text)
        
        # In your query_weaviate function, modify the near_vector call:
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=2,
            return_metadata=MetadataQuery(distance=True),  # Use the imported class directly
            return_properties=["title", "content", "metadata"]
        )
        
        # Get the objects from the response
        results = response.objects
        
        # Print raw metadata for debugging
        if hasattr(response, 'objects') and len(response.objects) > 0:
            first_result = response.objects[0]
            if hasattr(first_result, 'metadata'):
                print("DEBUG: First result metadata values:")
                for attr in ['distance', 'certainty', 'score']:
                    try:
                        value = getattr(first_result.metadata, attr)
                        print(f"  {attr}: {value}")
                    except AttributeError:
                        print(f"  {attr}: Not available")
        
        return results
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        return []
    finally:
        if 'client' in locals():
            client.close()  # Close the connection

if __name__ == "__main__":
    query_text = input("Введите запрос: ")
    results = query_weaviate(query_text)
    
    if results:
        print("Результаты поиска:")
        
        # Sort results by exact component match first
        sorted_results = []
        exact_matches = []
        other_matches = []
        
        for result in results:
            metadata_str = result.properties.get('metadata', '{}')
            try:
                import json
                metadata = json.loads(metadata_str)
                component = metadata.get('component', '')
                
                # Check if component exactly matches the query
                if query_text.lower() == component.lower():
                    exact_matches.append(result)
                else:
                    other_matches.append(result)
            except:
                other_matches.append(result)
        
        # Show exact matches first, then others
        sorted_results = exact_matches + other_matches
        
        for result in sorted_results:
            metadata_str = result.properties.get('metadata', '{}')
            try:
                import json
                metadata = json.loads(metadata_str)
                component = metadata.get('component', '')
                
                # Highlight exact matches
                if query_text.lower() == component.lower():
                    print(f"Title: {result.properties.get('title', 'No title')} [EXACT MATCH]")
                else:
                    print(f"Title: {result.properties.get('title', 'No title')}")
                
                if 'component' in metadata:
                    print(f"Component: {metadata.get('component')}")
                if 'version' in metadata:
                    print(f"Version: {metadata.get('version')}")
                
                # Print distance score from Weaviate v4
                if hasattr(result, 'metadata'):
                    # Access the distance attribute directly
                    try:
                        meta = result.metadata
                        print(f"Metadata: {meta}")
                    except AttributeError:
                        pass
                        
                    # Try accessing as dictionary
                    try:
                        if hasattr(result.metadata, '__dict__'):
                            metadata_dict = result.metadata.__dict__
                            if 'distance' in metadata_dict:
                                print(f"Distance: {metadata_dict['distance']}")
                            elif 'certainty' in metadata_dict:
                                print(f"Certainty: {metadata_dict['certainty']}")
                            elif 'score' in metadata_dict:
                                print(f"Score: {metadata_dict['score']}")
                    except:
                        pass
            except:
                print(f"Title: {result.properties.get('title', 'No title')}")
                
            # Show a snippet of content
            content = result.properties.get('content', '')
            if content:
                snippet = content[:200] + '...' if len(content) > 200 else content
                print(f"Content snippet: {snippet}")
                
            print("---")
    else:
        print("Нет результатов или произошла ошибка.")
