import os
import json
import torch
import weaviate
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.config import Property, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()


class PDFFolderLoader:
    """
    A class to load PDF files from a folder, process them with Docling,
    and store their content as embeddings in Weaviate vector database.
    """
    
    def __init__(self, pdf_folder_path: str, weaviate_url: str = None):
        """
        Initialize the PDF folder loader.
        
        Args:
            pdf_folder_path: Path to the folder containing PDF files
            weaviate_url: URL of the Weaviate instance (optional, uses .env if not provided)
        """
        self.pdf_folder_path = Path(pdf_folder_path)
        self.weaviate_url = weaviate_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.collection_name = os.getenv("WEAVIATE_COLLECTION", "Document")
        
        # Initialize Docling converter with PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
        pipeline_options.do_table_structure = True  # Extract table structure
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Initialize embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        # Connect to Weaviate
        # Use simple connection to local instance
        self.client = weaviate.connect_to_local()
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists in Weaviate."""
        try:
            if not self.client.collections.exists(self.collection_name):
                self.client.collections.create(
                    self.collection_name,
                    description="PDF documents with content and metadata",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Document content chunk"),
                        Property(name="title", data_type=DataType.TEXT, description="Document title"),
                        Property(name="source", data_type=DataType.TEXT, description="Source PDF file path"),
                        Property(name="page_number", data_type=DataType.INT, description="Page number"),
                        Property(name="chunk_index", data_type=DataType.INT, description="Chunk index within document"),
                        Property(name="metadata", data_type=DataType.TEXT, description="Additional metadata in JSON format")
                    ]
                )
                print(f"✅ Collection '{self.collection_name}' created")
            else:
                print(f"ℹ️ Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        return embeddings.tolist()
    
    def _process_pdf_file(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single PDF file using Docling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            print(f"📄 Processing: {pdf_path.name}")
            
            # Convert PDF using Docling
            result = self.converter.convert(pdf_path)
            
            # Extract text content
            full_text = result.document.export_to_markdown()
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Prepare document chunks
            document_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    chunk_data = {
                        "content": chunk,
                        "title": pdf_path.stem,
                        "source": str(pdf_path),
                        "page_number": 0,  # Docling doesn't provide page-level splitting by default
                        "chunk_index": i,
                        "metadata": {
                            "file_name": pdf_path.name,
                            "file_size": pdf_path.stat().st_size,
                            "processing_timestamp": str(datetime.now()),
                            "total_chunks": len(chunks)
                        }
                    }
                    document_chunks.append(chunk_data)
            
            print(f"✅ Processed {pdf_path.name}: {len(document_chunks)} chunks")
            return document_chunks
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            return []
    
    def _store_chunks_in_weaviate(self, chunks: List[Dict[str, Any]]):
        """
        Store document chunks in Weaviate with embeddings.
        
        Args:
            chunks: List of document chunks to store
        """
        try:
            docs_collection = self.client.collections.get(self.collection_name)
            
            for chunk in chunks:
                # Generate embedding for the chunk content
                embedding = self._generate_embedding(chunk["content"])
                
                # Store in Weaviate
                docs_collection.data.insert(
                    properties={
                        "content": chunk["content"],
                        "title": chunk["title"],
                        "source": chunk["source"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "metadata": json.dumps(chunk["metadata"], ensure_ascii=False)
                    },
                    vector=embedding
                )
            
            print(f"✅ Stored {len(chunks)} chunks in Weaviate")
            
        except Exception as e:
            print(f"❌ Error storing chunks in Weaviate: {e}")
    
    def load_pdf_folder(self) -> int:
        """
        Load all PDF files from the specified folder and store them in Weaviate.
        
        Returns:
            Number of successfully processed PDF files
        """
        if not self.pdf_folder_path.exists():
            print(f"❌ PDF folder does not exist: {self.pdf_folder_path}")
            return 0
        
        # Find all PDF files in the folder
        pdf_files = list(self.pdf_folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in: {self.pdf_folder_path}")
            return 0
        
        print(f"📁 Found {len(pdf_files)} PDF files to process")
        
        processed_count = 0
        total_chunks = 0
        
        for pdf_file in pdf_files:
            try:
                # Process PDF file
                chunks = self._process_pdf_file(pdf_file)
                
                if chunks:
                    # Store chunks in Weaviate
                    self._store_chunks_in_weaviate(chunks)
                    processed_count += 1
                    total_chunks += len(chunks)
                
            except Exception as e:
                print(f"❌ Failed to process {pdf_file.name}: {e}")
        
        print(f"🎉 Processing complete!")
        print(f"📊 Successfully processed: {processed_count}/{len(pdf_files)} PDF files")
        print(f"📊 Total chunks stored: {total_chunks}")
        
        return processed_count
    
    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
            print("🔌 Weaviate connection closed")


def main():
    """Main function to run the PDF folder loader."""
    # Configuration
    pdf_folder = os.path.normpath("../Files/pdf")  # Adjust path as needed
    
    # Create PDF folder if it doesn't exist
    os.makedirs(pdf_folder, exist_ok=True)
    
    try:
        # Initialize and run the PDF loader
        loader = PDFFolderLoader(pdf_folder)
        
        # Check if Weaviate is ready
        if not loader.client.is_ready():
            print("❌ Weaviate is not ready. Please ensure Weaviate is running on localhost:50051.")
            return
        
        print("✅ Connected to Weaviate")
        
        # Load PDF files
        processed_files = loader.load_pdf_folder()
        
        if processed_files == 0:
            print("ℹ️ No PDF files were processed. Please add PDF files to the folder:")
            print(f"   {os.path.abspath(pdf_folder)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Always close the connection
        if 'loader' in locals():
            loader.close()


if __name__ == "__main__":
    main()