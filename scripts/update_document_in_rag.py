import weaviate
import json
import uuid
import sys
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

def update_document_in_rag(json_file_path, component_name=None):
    """
    Update a document in the RAG system based on component name or create a new one if it doesn't exist
    
    Args:
        json_file_path: Path to the JSON file with the updated document
        component_name: Name of the component to update (if None, will be extracted from the JSON)
    """
    # Load the document to update
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            updated_doc = json.load(f)
        
        if not isinstance(updated_doc, dict):
            if isinstance(updated_doc, list) and len(updated_doc) > 0:
                updated_doc = updated_doc[0]  # Take the first document if it's a list
            else:
                print("Error: Invalid JSON format. Expected an object or array of objects.")
                return
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract component name if not provided
    if component_name is None:
        if 'metadata' in updated_doc and 'component' in updated_doc['metadata']:
            component_name = updated_doc['metadata']['component']
        else:
            print("Error: Component name not provided and not found in document metadata")
            return
    
    print(f"Updating document for component: {component_name}")
    
    # Load models for embedding
    print("Loading embedding models...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    client = weaviate.connect_to_local()
    
    try:
        # Check if Document collection exists
        collections = client.collections.list_all()
        if "Document" not in collections:
            print("Error: Document collection not found in Weaviate")
            return
        
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Search for existing document with the same component
        print(f"Searching for existing document with component: {component_name}")
        
        # Query to find documents with matching component in metadata
        query_results = collection.query.fetch_objects(
            limit=10  # Get multiple results to ensure we find the right one
        )
        
        # Find the document with matching component
        existing_doc = None
        for obj in query_results.objects:
            try:
                # Parse metadata which is stored as a JSON string
                metadata = json.loads(obj.properties.get('metadata', '{}'))
                if metadata.get('component') == component_name:
                    existing_doc = obj
                    break
            except:
                continue
        
        # Generate embedding for the updated content
        print("Generating embedding for updated content...")
        embedding = generate_embedding(updated_doc['content'], model, tokenizer)
        
        if existing_doc:
            # Update existing document
            print(f"Updating existing document with UUID: {existing_doc.uuid}")
            
            # Update the document
            collection.data.update(
                uuid=existing_doc.uuid,
                properties={
                    "content": updated_doc['content'],
                    "title": updated_doc['metadata'].get('title', ''),
                    "metadata": json.dumps(updated_doc['metadata'], ensure_ascii=False)
                },
                vector=embedding  # Update the vector with new embedding
            )
            print("✅ Document updated successfully")
        else:
            # Create new document
            print("No existing document found. Creating new document...")
            
            # Generate UUID if not provided
            doc_uuid = updated_doc.get('id', str(uuid.uuid4()))
            
            # Insert new document
            collection.data.insert(
                uuid=doc_uuid,
                properties={
                    "content": updated_doc['content'],
                    "title": updated_doc['metadata'].get('title', ''),
                    "metadata": json.dumps(updated_doc['metadata'], ensure_ascii=False)
                },
                vector=embedding
            )
            print(f"✅ New document created with UUID: {doc_uuid}")
        
    except Exception as e:
        print(f"Error updating document: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_document_in_rag.py <json_file_path> [component_name]")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    component_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    update_document_in_rag(json_file_path, component_name)