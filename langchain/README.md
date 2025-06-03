# RAG Chat Application with DeepSeek & Weaviate

A sophisticated Retrieval-Augmented Generation (RAG) chat application that combines DeepSeek LLM with Weaviate vector database for intelligent document-based conversations.

## Features

- 🤖 **DeepSeek LLM Integration**: Powered by DeepSeek's advanced language model
- 🔍 **Vector Search**: Uses Weaviate for efficient semantic document retrieval
- 💬 **Interactive Chat**: Streamlit-based chat interface with conversation history
- 📚 **Source Citations**: Shows relevant document sources for each response
- ⚙️ **Configurable**: Adjustable search parameters and chunk sizes
- 🔒 **Secure**: Environment-based API key management

## Prerequisites

1. **Python 3.8+**
2. **Weaviate Database**: Running locally or remotely
3. **DeepSeek API Key**: Get one from [DeepSeek Platform](https://platform.deepseek.com/)

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys and configuration
   ```

3. **Set up Weaviate:**
   - For local setup with Docker:
     ```bash
     docker run -d \
       --name weaviate \
       -p 8080:8080 \
       -e QUERY_DEFAULTS_LIMIT=25 \
       -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
       -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
       semitechnologies/weaviate:latest
     ```

## Usage

### Quick Start

Run the application using the provided script:

```bash
python run_app.py
```

Or directly with Streamlit:

```bash
streamlit run app_weaviate.py
```

### Configuration

The application can be configured through the sidebar:

- **DeepSeek API Key**: Your DeepSeek API key
- **Weaviate URL**: URL of your Weaviate instance (default: http://localhost:8080)
- **Collection Name**: Name of the Weaviate collection containing your documents
- **Search Parameters**: Adjust search results limit and text chunk size

### Environment Variables

Create a `.env` file with the following variables:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
WEAVIATE_URL=http://localhost:8080
WEAVIATE_COLLECTION=Document
```

## Document Ingestion

Before using the chat application, you need to ingest documents into Weaviate. You can use the existing scripts in the parent directory:

- `embed_and_store2weaviate.py`: For embedding and storing documents
- `convert_docx2text.py`: For converting DOCX files to text
- Other utility scripts for document processing

## Application Architecture

### Components

1. **Frontend**: Streamlit-based chat interface
2. **LLM**: DeepSeek chat model via OpenAI-compatible API
3. **Embeddings**: HuggingFace sentence-transformers
4. **Vector Database**: Weaviate for document storage and retrieval
5. **RAG Chain**: LangChain for orchestrating retrieval and generation

### Key Features

- **Semantic Search**: Uses vector embeddings for finding relevant documents
- **Context-Aware Responses**: Combines retrieved documents with user queries
- **Source Attribution**: Shows which documents were used for each response
- **Chat History**: Maintains conversation context within the session
- **Error Handling**: Robust error handling for API and database connections

## Troubleshooting

### Common Issues

1. **Weaviate Connection Failed**
   - Ensure Weaviate is running on the specified URL
   - Check if the collection exists and contains documents

2. **DeepSeek API Errors**
   - Verify your API key is correct and has sufficient credits
   - Check the API endpoint is accessible

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt` to install all required packages

4. **Empty Responses**
   - Ensure your Weaviate collection contains embedded documents
   - Check if the collection name matches your configuration

### Performance Tips

- Adjust the search limit based on your needs (fewer results = faster responses)
- Use appropriate chunk sizes for your document types
- Consider using GPU acceleration for embeddings if available

## Development

### File Structure

```
langchain/
├── app_weaviate.py      # Main application
├── requirements.txt     # Python dependencies
├── run_app.py          # Application launcher
├── .env.example        # Environment template
└── README.md           # This file
```

### Extending the Application

- **Custom Embeddings**: Replace HuggingFace embeddings with other models
- **Different LLMs**: Swap DeepSeek with other OpenAI-compatible models
- **Enhanced UI**: Add more Streamlit components for better user experience
- **Document Upload**: Add functionality to upload and process documents directly

## License

This project is part of the Search-Document repository and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the existing scripts in the parent directory
3. Ensure all prerequisites are properly configured