import weaviate
import json
import sys

def read_weaviate_schema():
    """
    Read and display the Weaviate schema using v4 API
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        print("=== WEAVIATE SCHEMA ===")
        
        # Get all collections
        collections = client.collections.list_all()
        print(f"Collections: {list(collections.keys())}")
        
        # Iterate through each collection
        for collection_name in collections:
            collection = client.collections.get(collection_name)
            print(f"\n== Collection: {collection.name} ==")
            
            # Get properties
            try:
                # Get properties using the collection object
                properties = []
                
                # Try to get properties from the collection
                if hasattr(collection, 'properties'):
                    if hasattr(collection.properties, 'get'):
                        properties = collection.properties.get()
                    else:
                        properties = collection.properties
                
                if properties:
                    print(f"\nProperties:")
                    for prop in properties:
                        if hasattr(prop, 'name') and hasattr(prop, 'data_type'):
                            print(f"- {prop.name}: {prop.data_type}")
                            if hasattr(prop, 'description') and prop.description:
                                print(f"  Description: {prop.description}")
                        elif isinstance(prop, dict):
                            print(f"- {prop.get('name')}: {prop.get('dataType')}")
                            if prop.get('description'):
                                print(f"  Description: {prop.get('description')}")
                else:
                    print("\nNo properties found or could not access properties")
            except Exception as e:
                print(f"Error getting properties: {e}")
            
            # Get object count
            try:
                # Try to get a sample object to count
                objects = collection.query.fetch_objects(limit=1)
                if hasattr(objects, 'total'):
                    print(f"\nTotal objects: {objects.total}")
                else:
                    print("\nCould not determine object count")
            except Exception as e:
                print(f"Error getting object count: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

def read_document_by_component(component_name):
    """
    Read a document from Weaviate by component name
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        print(f"=== SEARCHING FOR DOCUMENT WITH COMPONENT: {component_name} ===")
        
        # Get the Document collection
        collection = client.collections.get("Document")
        
        # Query for documents
        results = collection.query.fetch_objects(
            limit=10  # Get multiple results to ensure we find the right one
        )
        
        # Find the document with matching component
        found = False
        for obj in results.objects:
            try:
                # Parse metadata which is stored as a JSON string
                metadata = json.loads(obj.properties.get('metadata', '{}'))
                if metadata.get('component') == component_name:
                    found = True
                    print(f"\n== Document Found ==")
                    print(f"UUID: {obj.uuid}")
                    print(f"Title: {obj.properties.get('title')}")
                    print(f"Component: {metadata.get('component')}")
                    print(f"Version: {metadata.get('version')}")
                    print(f"Date: {metadata.get('date')}")
                    
                    # Print content (truncated)
                    content = obj.properties.get('content', '')
                    if len(content) > 200:
                        print(f"\nContent (first 200 chars): {content[:200]}...")
                    else:
                        print(f"\nContent: {content}")
                    
                    break
            except:
                continue
        
        if not found:
            print(f"No document found with component: {component_name}")
            print("Available components:")
            
            # List available components
            components = set()
            for obj in results.objects:
                try:
                    metadata = json.loads(obj.properties.get('metadata', '{}'))
                    if 'component' in metadata:
                        components.add(metadata.get('component'))
                except:
                    continue
            
            for component in components:
                print(f"- {component}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, show schema
        read_weaviate_schema()
    else:
        # Argument provided, search for document by component
        component_name = sys.argv[1]
        read_document_by_component(component_name)