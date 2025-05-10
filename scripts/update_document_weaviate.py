import weaviate
import json
import uuid
from datetime import datetime

def update_rag_document(updated_doc_path, collection_name="Documentation"):
    """
    Update a document in the RAG system or create it if it doesn't exist.
    
    Args:
        updated_doc_path: Path to the JSON file with updated document
        collection_name: Name of the collection in the vector database
    """
    # 1. Load the updated document
    with open(updated_doc_path, 'r', encoding='utf-8') as f:
        updated_doc = json.load(f)
    
    # 2. Extract key information
    component = updated_doc["metadata"]["component"]
    document_key = f"{component}_spec"  # Create a stable identifier
    
    # 3. Connect to vector database
    client = weaviate.connect_to_local()
    collection = client.collections.get(collection_name)
    
    # 4. Check if document exists
    query_result = collection.query.fetch_objects(
        filters={
            "path": ["metadata", "document_key"],
            "operator": "Equal",
            "valueText": document_key
        }
    )
    
    # 5. Update timestamp and other metadata
    updated_doc["metadata"]["last_updated"] = datetime.utcnow().isoformat()
    
    if query_result.objects:
        # Document exists, update it
        existing_doc = query_result.objects[0]
        existing_id = existing_doc.uuid
        
        # Increment revision
        current_revision = existing_doc.properties["metadata"].get("revision", 0)
        updated_doc["metadata"]["revision"] = current_revision + 1
        
        # Update the document
        collection.data.update(
            uuid=existing_id,
            properties={
                "content": updated_doc["content"],
                "metadata": updated_doc["metadata"]
            }
        )
        print(f"Document '{document_key}' updated successfully (ID: {existing_id}, Revision: {current_revision + 1})")
    else:
        # Document doesn't exist, create new
        new_id = updated_doc.get("id", str(uuid.uuid4()))
        
        # Set initial revision
        updated_doc["metadata"]["revision"] = 1
        updated_doc["metadata"]["document_key"] = document_key
        
        # Create new document
        collection.data.insert(
            uuid=new_id,
            properties={
                "content": updated_doc["content"],
                "metadata": updated_doc["metadata"]
            }
        )
        print(f"New document '{document_key}' created (ID: {new_id}, Revision: 1)")
    
    # 6. Validate the update with a test query
    validation = collection.query.near_text(
        query=component,
        limit=1
    )
    
    if validation.objects:
        print("Validation successful: Document is retrievable in vector search")
    else:
        print("Warning: Document may not be properly indexed")

# Example usage
update_rag_document("updated_document.json")
