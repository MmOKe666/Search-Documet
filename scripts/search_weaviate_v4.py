import weaviate
import json
import argparse
from transformers import AutoTokenizer, AutoModel
import torch

def generate_embedding(text, model, tokenizer):
    """Generate embedding for text using the provided model"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                      padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    return embeddings.tolist()

def vector_search(query_text, limit=5):
    """
    Search documents using vector search by generating embeddings locally
    """
    print(f"=== VECTOR SEARCHING DOCUMENTS WITH QUERY: '{query_text}' ===")
    
    # Load models for embedding
    print("Loading embedding models...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embedding for the query
    query_embedding = generate_embedding(query_text, model, tokenizer)
    
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Search using the generated vector
        # In v4.14.1, the parameter is named 'near_vector' not 'vector'
        response = collection.query.near_vector(
            near_vector=query_embedding,  # Correct parameter name
            limit=limit,
            return_properties=["title", "content", "metadata"]
        )
        
        # Get the objects from the response
        results = response.objects
        
        if results:
            print(f"\nFound {len(results)} results:")
            
            for i, result in enumerate(results):
                print(f"\n== Result {i+1} ==")
                print(f"Title: {result.properties.get('title')}")
                print(f"Distance: {result.metadata.distance}")
                
                # Parse metadata
                try:
                    metadata = json.loads(result.properties.get('metadata', '{}'))
                    print(f"Component: {metadata.get('component')}")
                    print(f"Version: {metadata.get('version')}")
                except:
                    print("Could not parse metadata")
                
                # Print content snippet
                content = result.properties.get('content', '')
                if len(content) > 200:
                    print(f"Content snippet: {content[:200]}...")
                else:
                    print(f"Content: {content}")
                
                print("---")
        else:
            print("No results found")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

def keyword_search(query_text, limit=5):
    """
    Search documents using BM25 search (keyword-based)
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        print(f"=== KEYWORD SEARCHING DOCUMENTS WITH QUERY: '{query_text}' ===")
        
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Use BM25 search
        response = collection.query.bm25(
            query=query_text,
            limit=limit,
            return_properties=["title", "content", "metadata"]
        )
        
        # Get the objects from the response
        results = response.objects
        
        if results:
            print(f"\nFound {len(results)} results:")
            
            for i, result in enumerate(results):
                print(f"\n== Result {i+1} ==")
                print(f"Title: {result.properties.get('title')}")
                if hasattr(result.metadata, 'score'):
                    print(f"Score: {result.metadata.score}")
                
                # Parse metadata
                try:
                    metadata = json.loads(result.properties.get('metadata', '{}'))
                    print(f"Component: {metadata.get('component')}")
                    print(f"Version: {metadata.get('version')}")
                except:
                    print("Could not parse metadata")
                
                # Print content snippet
                content = result.properties.get('content', '')
                if len(content) > 200:
                    print(f"Content snippet: {content[:200]}...")
                else:
                    print(f"Content: {content}")
                
                print("---")
        else:
            print("No results found")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Weaviate documents")
    parser.add_argument("query", help="Query text for search")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")
    parser.add_argument("--method", choices=["vector", "keyword"], default="vector", 
                        help="Search method: vector (semantic) or keyword (BM25)")
    
    args = parser.parse_args()
    
    if args.method == "vector":
        vector_search(args.query, args.limit)
    else:
        keyword_search(args.query, args.limit)