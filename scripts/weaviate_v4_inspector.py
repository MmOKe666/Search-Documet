import weaviate
import json
import sys

def inspect_weaviate():
    """
    A Weaviate schema inspector that uses only the public v4 API
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        print("=== WEAVIATE V4 SCHEMA INSPECTOR ===")
        
        # 1. Get collections
        collections = client.collections.list_all()
        print(f"\nNumber of collections: {len(collections)}")
        
        # 2. List all collections
        print("\n== Collections ==")
        for name in collections:
            print(f"- {name}")
        
        # 3. If a collection name is provided as argument, show details
        collection_name = "Document"  # Default
        if len(sys.argv) > 1:
            collection_name = sys.argv[1]
        
        # Check if collection exists
        if collection_name in collections:
            print(f"\n== Details for Collection: {collection_name} ==")
            
            # Get collection
            collection = client.collections.get(collection_name)
            
            # Print basic info
            print(f"\nName: {collection.name}")
            
            # Get properties
            try:
                # Try different approaches to get properties
                properties = []
                
                # Approach 1: Try properties attribute
                if hasattr(collection, 'properties'):
                    if callable(getattr(collection.properties, 'get', None)):
                        properties = collection.properties.get()
                    else:
                        properties = collection.properties
                
                # Approach 2: Try config.properties
                elif hasattr(collection, 'config') and hasattr(collection.config, 'properties'):
                    properties = collection.config.properties
                
                if properties:
                    print(f"\nProperties ({len(properties)}):")
                    for prop in properties:
                        if hasattr(prop, 'name') and hasattr(prop, 'data_type'):
                            print(f"- {prop.name}: {prop.data_type}")
                        elif isinstance(prop, dict):
                            print(f"- {prop.get('name')}: {prop.get('dataType')}")
                else:
                    print("\nCould not retrieve properties using available methods")
            except Exception as e:
                print(f"Error getting properties: {e}")
            
            # Get objects
            try:
                # Try to get objects
                objects = collection.query.fetch_objects(limit=1)
                
                if hasattr(objects, 'objects'):
                    # New API style
                    if objects.objects:
                        print("\nSample object:")
                        obj = objects.objects[0]
                        print(f"UUID: {obj.uuid}")
                        
                        # Print properties
                        print("Properties:")
                        for key, value in obj.properties.items():
                            if isinstance(value, str) and len(value) > 100:
                                print(f"- {key}: {value[:100]}... (truncated)")
                            else:
                                print(f"- {key}: {value}")
                    else:
                        print("\nNo objects found in collection")
                else:
                    # Try to access as dictionary
                    print("\nObjects returned in non-standard format")
                    print(f"Type: {type(objects)}")
                    if hasattr(objects, '__dict__'):
                        print(f"Attributes: {dir(objects)}")
            except Exception as e:
                print(f"Error getting objects: {e}")
                import traceback
                traceback.print_exc()
            
            # Try to get count
            try:
                # Try different approaches for count
                count = None
                
                # Approach 1: Try aggregate().objects_count
                if hasattr(collection, 'aggregate') and hasattr(collection.aggregate, 'over_all'):
                    count_result = collection.aggregate.over_all()
                    if hasattr(count_result, 'objects_count'):
                        count = count_result.objects_count
                
                if count is not None:
                    print(f"\nTotal objects: {count}")
                else:
                    print("\nCould not retrieve object count using available methods")
            except Exception as e:
                print(f"Error getting count: {e}")
        else:
            print(f"\nCollection '{collection_name}' not found.")
            print(f"Available collections: {list(collections.keys())}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    inspect_weaviate()