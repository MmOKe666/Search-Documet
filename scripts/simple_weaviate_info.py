import weaviate

def print_weaviate_info():
    """
    A very simple Weaviate info printer using only public v4 API methods
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local()
    
    try:
        print("=== WEAVIATE INFO ===")
        
        # Get collections
        collections = client.collections.list_all()
        print(f"Collections: {list(collections.keys())}")
        
        # Get info about Document collection
        collection_name = "Document"
        if collection_name in collections:
            print(f"\n== Collection: {collection_name} ==")
            collection = client.collections.get(collection_name)
            
            # Basic info
            print(f"Name: {collection.name}")
            
            # Try to get a sample object
            try:
                objects = collection.query.fetch_objects(limit=1)
                
                if objects and hasattr(objects, 'objects') and objects.objects:
                    obj = objects.objects[0]
                    print(f"\nSample object UUID: {obj.uuid}")
                    
                    # Print properties
                    print("Properties:")
                    for key, value in obj.properties.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"- {key}: {value[:100]}... (truncated)")
                        else:
                            print(f"- {key}: {value}")
                else:
                    print("\nNo objects found")
            except Exception as e:
                print(f"Error getting objects: {e}")
        else:
            print(f"Collection '{collection_name}' not found")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    print_weaviate_info()