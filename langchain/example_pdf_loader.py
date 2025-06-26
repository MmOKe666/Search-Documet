#!/usr/bin/env python3
"""
Example script demonstrating how to use the PDFFolderLoader class.
"""

import os
from pdfFolderLoader import PDFFolderLoader


def main():
    """Example usage of PDFFolderLoader."""
    
    # Configuration
    pdf_folder_path = "../Files/pdf"  # Adjust this path as needed
    
    print("🚀 PDF Folder Loader Example")
    print("=" * 50)
    
    # Create the PDF folder if it doesn't exist
    os.makedirs(pdf_folder_path, exist_ok=True)
    print(f"📁 PDF folder: {os.path.abspath(pdf_folder_path)}")
    
    try:
        # Initialize the PDF loader
        print("\n🔧 Initializing PDF Folder Loader...")
        loader = PDFFolderLoader(pdf_folder_path)
        
        # Check Weaviate connection
        if not loader.client.is_ready():
            print("❌ Weaviate is not ready!")
            print("Please ensure Weaviate is running on http://localhost:8080")
            print("You can start it with: docker-compose up -d")
            return
        
        print("✅ Connected to Weaviate successfully")
        
        # Load and process PDF files
        print("\n📚 Starting PDF processing...")
        processed_count = loader.load_pdf_folder()
        
        if processed_count > 0:
            print(f"\n🎉 Successfully processed {processed_count} PDF files!")
            print("📊 Documents are now available in Weaviate for search and retrieval.")
        else:
            print(f"\n⚠️ No PDF files found in {pdf_folder_path}")
            print("Please add some PDF files to the folder and run again.")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        
    finally:
        # Clean up
        if 'loader' in locals():
            loader.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()