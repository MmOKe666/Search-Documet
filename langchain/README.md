# RAG Chat Application with OpenRouter & Weaviate

A sophisticated Retrieval-Augmented Generation (RAG) chat application that combines multiple LLMs via OpenRouter with Weaviate vector database for intelligent document-based conversations.

## Features

- 🤖 **Multi-LLM Support**: OpenRouter integration with GPT-4, Claude, DeepSeek, Gemini, and Llama models
- 🔍 **Vector Search**: Uses Weaviate for efficient semantic document retrieval
- 💬 **Interactive Chat**: Streamlit-based chat interface with conversation history
- 📚 **Source Citations**: Shows relevant document sources with similarity scores
- ⚙️ **Configurable**: Adjustable search parameters and chunk sizes
- 🔒 **Secure**: Environment-based configuration management
- ✅ **Fixed Issues**: Resolved ScriptRunContext warnings and metadata handling errors

## Prerequisites

1. **Python 3.8+**
2. **Weaviate Database**: Running locally or remotely
3. **OpenRouter API Key**: Get one from [OpenRouter Platform](https://openrouter.ai/)

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
streamlit run LangGraph_app.py
```

### App Management Commands

**Start the app:**
```bash
# Activate virtual environment and start
source ../3.12-env/bin/activate
streamlit run LangGraph_app.py --server.port 8501

# Or run in background (headless)
streamlit run LangGraph_app.py --server.port 8501 --server.headless true &
```

**Stop the app:**
```bash
# Find the process ID
ps aux | grep streamlit | grep -v grep

# Kill the process (replace XXXX with actual PID)
kill XXXX

# Or kill all streamlit processes
pkill -f streamlit
```

**Restart the app:**
```bash
# Stop and start in one command
pkill -f streamlit && sleep 2 && source ../3.12-env/bin/activate && streamlit run LangGraph_app.py --server.port 8501 --server.headless true &
```

**Check if app is running:**
```bash
# Check process
ps aux | grep streamlit | grep -v grep

# Check if port is accessible
curl -s http://localhost:8501 | head -5
```

### Configuration

The application can be configured through the sidebar:

- **OpenRouter API Key**: Your OpenRouter API key
- **Model Selection**: Choose from GPT-4, Claude, DeepSeek, Gemini, or Llama models
- **Weaviate URL**: URL of your Weaviate instance (default from environment)
- **Collection Name**: Name of the Weaviate collection containing your documents
- **Search Parameters**: Adjust search results limit and text chunk size

### Environment Variables

**⚠️ Important: You must configure your API key before using the application.**

1. **Get OpenRouter API Key:**
   - Go to [OpenRouter](https://openrouter.ai/)
   - Create an account and generate an API key

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env file and replace 'your_openrouter_api_key_here' with your actual API key
   ```

3. **Environment Variables:**
   ```env
   # OpenRouter API Configuration (REQUIRED)
   OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
   
   # Weaviate Configuration
   WEAVIATE_URL=http://localhost:50051
   WEAVIATE_COLLECTION=Document
   
   # Optional: Other API Keys
   GOOGLE_API_KEY=your_google_api_key_here
   ```

**Alternative:** You can also enter the API key directly in the app's sidebar.

## Document Ingestion

Before using the chat application, you need to ingest documents into Weaviate. You can use the existing scripts in the parent directory:

- `embed_and_store2weaviate.py`: For embedding and storing documents
- `convert_docx2text.py`: For converting DOCX files to text
- Other utility scripts for document processing

## Application Architecture

### Components

1. **Frontend**: Streamlit-based chat interface
2. **LLM**: Multiple models via OpenRouter (GPT-4, Claude, DeepSeek, etc.)
3. **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
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

1. **Authentication Error (401)**
   - **Cause:** Missing or invalid OpenRouter API key
   - **Solution:** 
     - Get API key from [OpenRouter](https://openrouter.ai/)
     - Add it to `.env` file or sidebar
     - Restart the application

2. **Weaviate Connection Failed**
   - Ensure Weaviate is running on the specified URL
   - Check if the collection exists and contains documents

3. **OpenRouter API Errors**
   - Verify your API key is correct and has sufficient credits
   - Check the selected model is available and accessible

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

## Recent Fixes & Improvements

### Fixed Issues (Latest Update)

1. **ScriptRunContext Warnings**: 
   - Added proper session state initialization
   - Fixed `processed` dictionary initialization

2. **Metadata Assignment Error**: 
   - Fixed `'str' object does not support item assignment` error
   - Added type checking for Weaviate metadata handling
   - Ensured metadata is always a dictionary before assignment

3. **Environment Configuration**:
   - Moved all hardcoded URLs to environment variables
   - Added comprehensive `.env` configuration
   - Improved security with environment-based settings

4. **Streamlit Best Practices**:
   - Proper session state management
   - Improved error handling and user feedback
   - Better resource cleanup

### Technical Findings

- **Weaviate Metadata**: Can return strings instead of dictionaries, requiring type validation
- **Streamlit Context**: Proper session state initialization prevents ScriptRunContext warnings
- **LangChain Integration**: Works seamlessly with OpenRouter's OpenAI-compatible API
- **Performance**: HuggingFace embeddings provide good balance of speed and accuracy

### Extending the Application

- **Custom Embeddings**: Replace HuggingFace embeddings with other models
- **Additional Models**: OpenRouter supports 100+ models beyond the current selection
- **Enhanced UI**: Add file upload, document management, and advanced search filters
- **Multi-Collection**: Support multiple Weaviate collections for different document types

## License

This project is part of the Search-Document repository and follows the same licensing terms.

## Known Limitations

- Requires pre-indexed documents in Weaviate
- Limited to text-based documents
- Session state resets on browser refresh
- No built-in document upload functionality
- App needs manual restart after code changes

## Future Enhancements

- [ ] Direct document upload and processing
- [ ] Multi-collection support
- [ ] Advanced search filters
- [ ] Conversation export/import
- [ ] Custom embedding model selection
- [ ] Real-time document indexing

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the existing scripts in the parent directory
3. Ensure all prerequisites are properly configured
4. Verify environment variables are correctly set