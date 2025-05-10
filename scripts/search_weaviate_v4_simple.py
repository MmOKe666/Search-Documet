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

def simple_search(query_text, limit=5):
    """
    A very simple search that retrieves all documents and ranks them by keyword matching
    """
    print(f"=== SIMPLE SEARCHING DOCUMENTS WITH QUERY: '{query_text}' ===")
    
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Get all documents
        all_docs = collection.query.fetch_objects(
            limit=100,  # Adjust as needed
            return_properties=["title", "content", "metadata"]
        )
        
        if not all_docs.objects:
            print("No documents found in collection")
            return
        
        print(f"Retrieved {len(all_docs.objects)} documents for processing")
        
        # Simple keyword matching
        query_terms = query_text.lower().split()
        results = []
        
        for doc in all_docs.objects:
            content = doc.properties.get('content', '').lower()
            title = doc.properties.get('title', '').lower()
            
            # Count term occurrences
            term_matches = {}
            for term in query_terms:
                content_count = content.count(term)
                title_count = title.count(term) * 2  # Title matches count double
                term_matches[term] = content_count + title_count
            
            # Calculate score based on term frequency
            total_matches = sum(term_matches.values())
            matched_terms = sum(1 for count in term_matches.values() if count > 0)
            
            # Score is a combination of total matches and term coverage
            score = total_matches * (matched_terms / len(query_terms) if query_terms else 0)
            
            results.append({
                'doc': doc,
                'score': score,
                'matched_terms': matched_terms,
                'total_matches': total_matches
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top results
        top_results = results[:limit]
        
        if top_results:
            print(f"\nFound {len(top_results)} results:")
            
            for i, result in enumerate(top_results):
                doc = result['doc']
                print(f"\n== Result {i+1} ==")
                print(f"Title: {doc.properties.get('title')}")
                print(f"Score: {result['score']:.2f} (matched {result['matched_terms']}/{len(query_terms)} terms, {result['total_matches']} occurrences)")
                
                # Parse metadata
                try:
                    metadata = json.loads(doc.properties.get('metadata', '{}'))
                    print(f"Component: {metadata.get('component')}")
                    print(f"Version: {metadata.get('version')}")
                except:
                    print("Could not parse metadata")
                
                # Print content snippet with highlighted terms
                content = doc.properties.get('content', '')
                
                # Find a relevant snippet (first paragraph with query terms)
                paragraphs = content.split('\n\n')
                relevant_paragraph = None
                
                for para in paragraphs:
                    if any(term.lower() in para.lower() for term in query_terms):
                        relevant_paragraph = para
                        break
                
                if relevant_paragraph:
                    if len(relevant_paragraph) > 200:
                        print(f"Content snippet: {relevant_paragraph[:200]}...")
                    else:
                        print(f"Content snippet: {relevant_paragraph}")
                else:
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
    parser = argparse.ArgumentParser(description="Simple search for Weaviate documents")
    parser.add_argument("query", help="Query text for search")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")
    
    args = parser.parse_args()
    simple_search(args.query, args.limit)