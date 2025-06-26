# PDF Folder Loader with Docling and Weaviate

This module provides functionality to load PDF files from a folder, process them using the Docling library, and store their content as embeddings in a Weaviate vector database.

## Features

- **Docling Integration**: Uses Docling library for advanced PDF processing
- **OCR Support**: Automatically handles scanned PDFs with OCR
- **Table Structure Extraction**: Preserves table structures from PDFs
- **Text Chunking**: Intelligently splits documents into manageable chunks
- **Vector Embeddings**: Generates embeddings using sentence-transformers
- **Weaviate Storage**: Stores documents with metadata in Weaviate vector database

## Requirements

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `docling` - For PDF processing
- `weaviate-client` - For vector database operations
- `sentence-transformers` - For generating embeddings
- `langchain` - For text splitting utilities

## Setup

### 1. Configure Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit the `.env` file to set your configuration:

```env
# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_COLLECTION=Document

# Optional: API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Start Weaviate

Make sure Weaviate is running locally on port 8080. If you have Docker Compose configured:

```bash
docker-compose up -d
```

Or run Weaviate directly:

```bash
docker run -p 8080:8080 semitechnologies/weaviate:latest
```

### 2. Prepare PDF Files

Create a folder for your PDF files and add some PDFs:

```bash
mkdir -p ../Files/pdf
# Add your PDF files to this folder
```

## Usage

### Basic Usage

```python
from pdfFolderLoader import PDFFolderLoader

# Initialize the loader
loader = PDFFolderLoader("../Files/pdf")

# Process all PDF files in the folder
processed_count = loader.load_pdf_folder()

# Clean up
loader.close()
```

### Using the Example Script

Run the provided example script:

```bash
python example_pdf_loader.py
```

### Direct Execution

You can also run the module directly:

```bash
python pdfFolderLoader.py
```

## How It Works

1. **PDF Discovery**: Scans the specified folder for PDF files
2. **Docling Processing**: Each PDF is processed using Docling with:
   - OCR enabled for scanned documents
   - Table structure extraction
   - Markdown export for clean text
3. **Text Chunking**: Documents are split into chunks using RecursiveCharacterTextSplitter
4. **Embedding Generation**: Each chunk gets embedded using sentence-transformers
5. **Weaviate Storage**: Chunks are stored with metadata including:
   - Original content
   - Source file information
   - Page numbers and chunk indices
   - Processing timestamps

## Configuration

All configuration is now managed through environment variables in the `.env` file:

### Environment Variables

- **WEAVIATE_URL**: Weaviate server URL (default: `http://localhost:8080`)
- **WEAVIATE_COLLECTION**: Collection name for storing documents (default: `Document`)
- **DEEPSEEK_API_KEY**: Optional API key for DeepSeek services
- **GOOGLE_API_KEY**: Optional API key for Google services

### PDF Folder Path

Modify the `pdf_folder` variable in the main function:

```python
pdf_folder = os.path.normpath("../Files/pdf")  # Your PDF folder path
```

### Embedding Model

The default model is `sentence-transformers/all-MiniLM-L6-v2`. You can change it:

```python
self.tokenizer = AutoTokenizer.from_pretrained("your-model-name")
self.model = AutoModel.from_pretrained("your-model-name")
```

### Text Chunking Parameters

Adjust chunk size and overlap:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust chunk size
    chunk_overlap=200   # Adjust overlap
)
```

### Docling Pipeline Options

Customize PDF processing options:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # Enable/disable OCR
pipeline_options.do_table_structure = True  # Enable/disable table extraction
```

## Weaviate Schema

The loader creates a "Document" collection with the following properties:

- `content` (TEXT): The document chunk content
- `title` (TEXT): Document title (filename without extension)
- `source` (TEXT): Full path to the source PDF file
- `page_number` (INT): Page number (currently set to 0)
- `chunk_index` (INT): Index of the chunk within the document
- `metadata` (TEXT): Additional metadata in JSON format

## Troubleshooting

### Weaviate Connection Issues

- Ensure Weaviate is running on `http://localhost:8080` (or the URL specified in your `.env` file)
- Check if the port is available and not blocked by firewall
- Verify Docker container is running: `docker ps`
- Verify your `.env` file has the correct `WEAVIATE_URL` setting

### PDF Processing Issues

- Ensure PDF files are not corrupted
- Check file permissions for reading PDF files
- Large PDFs may take longer to process

### Memory Issues

- For large PDF collections, consider processing in batches
- Adjust chunk size to reduce memory usage
- Monitor system resources during processing

## Example Output

```
📁 Found 3 PDF files to process
📄 Processing: document1.pdf
✅ Processed document1.pdf: 15 chunks
✅ Stored 15 chunks in Weaviate
📄 Processing: document2.pdf
✅ Processed document2.pdf: 8 chunks
✅ Stored 8 chunks in Weaviate
📄 Processing: document3.pdf
✅ Processed document3.pdf: 12 chunks
✅ Stored 12 chunks in Weaviate
🎉 Processing complete!
📊 Successfully processed: 3/3 PDF files
📊 Total chunks stored: 35
```

## Next Steps

After loading your PDFs, you can:

1. Query the documents using Weaviate's vector search
2. Build a RAG (Retrieval-Augmented Generation) system
3. Create a search interface for your document collection
4. Integrate with LangChain for advanced document processing workflows

## Support

For issues related to:
- **Docling**: Check the [Docling documentation](https://github.com/DS4SD/docling)
- **Weaviate**: Check the [Weaviate documentation](https://weaviate.io/developers/weaviate)
- **LangChain**: Check the [LangChain documentation](https://python.langchain.com/)