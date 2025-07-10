
# inspired by https://github.com/AarohiSingla/Generative_AI/blob/main/L-8/gemini_rag_demo/app1.py
# author Nikolay ILYIN

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import weaviate
from weaviate.classes.query import MetadataQuery
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # You can set the level to INFO, WARNING, etc., as needed.
# Create a logger
logger = logging.getLogger(__name__)

# Import streamlit (missing import statement)
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chat with OpenRouter & Weaviate",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chat Application")
st.markdown("*Powered by OpenRouter API and Weaviate Vector Database*")

# Initialize session state variables properly
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = {}

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # OpenRouter API configuration
    openrouter_api_key = st.text_input(
        "OpenRouter API Key", 
        type="password", 
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Enter your OpenRouter API key"
    )
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-chat",
            "google/gemini-pro",
            "meta-llama/llama-3.1-8b-instruct"
        ],
        index=0,
        help="Choose the model to use for chat completions"
    )
    
    # Weaviate configuration
    weaviate_url = st.text_input(
        "Weaviate URL", 
        value=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        help="Weaviate instance URL"
    )
    
    collection_name = st.text_input(
        "Collection Name", 
        value=os.getenv("WEAVIATE_COLLECTION", "Document"),
        help="Name of the Weaviate collection"
    )
    
    # Search parameters
    search_limit = st.slider("Search Results Limit", 1, 10, 3)
    chunk_size = st.slider("Text Chunk Size", 500, 2000, 1000)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={'device': 'cpu'}
    )

# Initialize OpenRouter LLM
@st.cache_resource
def load_openrouter_llm(api_key, model):
    """Load and cache the OpenRouter LLM"""
    if not api_key or api_key == "your_openrouter_api_key_here":
        st.error("❌ Please provide a valid OpenRouter API key in the sidebar")
        st.info("💡 Get your API key from: https://openrouter.ai/")
        return None
    
    try:
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            temperature=0.1,
            max_tokens=2000
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# Weaviate connection and query functions
def connect_to_weaviate(url):
    """Connect to Weaviate instance"""
    try:
        if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
            client = weaviate.connect_to_local(host=url.split("://")[1].split(":")[0])
        else:
            client = weaviate.connect_to_custom(
                http_host=url.split("://")[1].split(":")[0],
                http_port=int(url.split(":")[-1]) if ":" in url.split("://")[1] else 80,
                http_secure=url.startswith("https")
            )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

def vectorize_text_with_model(text, embedding_model):
    """Vectorize text using the embedding model"""
    try:
        embeddings = embedding_model.embed_query(text)
        return embeddings
    except Exception as e:
        st.error(f"Error vectorizing text: {e}")
        return None

def query_weaviate(query_text, weaviate_url, collection_name, limit=3):
    """Query Weaviate for similar documents"""
    try:
        client = connect_to_weaviate(weaviate_url)
        if not client:
            return []
        
        # Check if collection exists
        if not client.collections.exists(collection_name):
            st.error(f"Collection '{collection_name}' does not exist in Weaviate")
            return []
        
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # Vectorize the query
        embedding_model = load_embedding_model()
        query_vector = vectorize_text_with_model(query_text, embedding_model)
        
        if not query_vector:
            return []
        
        # Perform vector search - return all properties to see what's available
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
            # Removed return_properties to get all available properties
        )
        
        # Convert results to LangChain documents
        documents = []
        for obj in response.objects:
            # Debug: log all available properties
            logger.debug(f"Available properties: {list(obj.properties.keys())}")
            
            # Use the content property directly from Weaviate
            doc_content = obj.properties.get("content", "")
            
            # Debug: log the actual content length and first 100 chars
            logger.debug(f"Content length: {len(doc_content)}, First 100 chars: {doc_content[:100]}")
            
            doc_title = (obj.properties.get("title") or 
                        obj.properties.get("name") or "Unknown")
            
            doc_metadata = obj.properties.get("metadata", {})
            
            # Validate content is not empty
            if not doc_content or not isinstance(doc_content, str):
                doc_content = "No content available"
            
            # Validate title
            if not doc_title or not isinstance(doc_title, str):
                doc_title = "Unknown"
            
            # Ensure metadata is a dictionary
            if not isinstance(doc_metadata, dict):
                doc_metadata = {}
            
            # Get distance from metadata
            distance = 0.0
            if hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                distance = float(obj.metadata.distance)
            
            # Create Document with validated fields
            doc = Document(
                page_content=str(doc_content),
                metadata={
                    "title": str(doc_title),
                    "source": str(doc_metadata.get("source", "Unknown")),
                    "distance": distance
                }
            )
            documents.append(doc)
            
            # Log document info without full content to avoid log spam
            logger.debug(f"Document added - Title: {doc_title}, Content length: {len(str(doc_content))}")
        
        return documents
        
    except Exception as e:
        st.error(f"Error querying Weaviate: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

# Create RAG chain
def create_simple_rag_chain(llm):
    """Create a simple RAG chain"""
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the question based on the provided context. If context does not contain the requested information, just say 'I don't have enough information to answer this question.'\n\n"
        "Context: {context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )
    
    # The | (pipe) operator creates a LangChain pipeline where data flows left to right:
    # prompt template -> LLM -> string output parser (converts LLM response to plain string)
    chain = prompt | llm | StrOutputParser()
    return chain



# Main chat interface
def main():
    # Load models
    embedding_model = load_embedding_model()
    llm = load_openrouter_llm(openrouter_api_key, model_name)
    
    if not llm:
        st.warning("⚠️ Please configure a valid OpenRouter API key to start chatting")
        st.info("📝 Steps to get started:")
        st.markdown("""
        1. Go to [OpenRouter](https://openrouter.ai/) and create an account
        2. Generate an API key
        3. Enter the API key in the sidebar
        4. Select a model and start chatting
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    # enumerate() returns (index, item) pairs starting from 1 for user-friendly numbering
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['title']}")
                        st.markdown(f"*Distance: {source.get('distance', 'N/A'):.4f}*")
                        # Show full content with toggle
                        if st.button(f"Show full content of Source {i}", key=f"hist_src_{i}_{len(st.session_state.messages)}"):
                            st.text(source['content'])
                        else:
                            content_preview = source['content'][:200] + ("..." if len(source['content']) > 200 else "")
                            st.markdown(f"```\n{content_preview}\n```")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Query Weaviate for relevant documents
                    relevant_docs = query_weaviate(
                        prompt, weaviate_url, collection_name, search_limit
                    )
                    
                    # Validate documents are proper Document objects
                    valid_docs = []
                    for doc in relevant_docs:
                        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                            valid_docs.append(doc)
                        else:
                            st.warning(f"Invalid document object: {type(doc)}")
                    
                    if not valid_docs:
                        response = "I couldn't find any relevant documents to answer your question. Please make sure your Weaviate database contains documents and is properly configured."
                        sources = []
                    else:
                        # Create simple RAG chain
                        qa_chain = create_simple_rag_chain(llm)
                        
                        # Generate response with validated documents
                        try:
                            context_parts = []
                            for doc in valid_docs:
                                if hasattr(doc, 'page_content') and doc.page_content:
                                    context_parts.append(str(doc.page_content))
                            
                            context = "\n\n".join(context_parts)
                            
                            if not context:
                                context = "No relevant content found."
                            
                            # Debug: Log the full context being sent to LLM
                            logger.debug(f"Full context sent to LLM (length: {len(context)}): {context}")
                            
                            # Use the simple chain with just context and input as strings
                            response = qa_chain.invoke({
                                "context": context,
                                "input": prompt
                            })
                        except Exception as ctx_error:
                            error_msg = str(ctx_error)
                            if "401" in error_msg or "auth" in error_msg.lower():
                                st.error("❌ Authentication failed. Please check your OpenRouter API key.")
                                st.info("💡 Make sure you have entered a valid API key in the sidebar.")
                            else:
                                st.error(f"Context generation error: {ctx_error}")
                            response = f"Error: Please check your API key configuration."
                            
                        # Prepare sources for display with validation
                        sources = []
                        try:
                            for doc in valid_docs:
                                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                                    sources.append({
                                        "title": str(doc.metadata.get("title", "Unknown")),
                                        "content": str(doc.page_content),
                                        "distance": float(doc.metadata.get("distance", 0.0))
                                    })
                        except Exception as src_error:
                            st.error(f"Source preparation error: {src_error}")
                            sources = []
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source['title']}")
                                st.markdown(f"*Distance: {source.get('distance', 'N/A'):.4f}*")
                                # Show full content with toggle
                                if st.button(f"Show full content of Source {i}", key=f"curr_src_{i}"):
                                    st.text(source['content'])
                                else:
                                    content_preview = source['content'][:200] + ("..." if len(source['content']) > 200 else "")
                                    st.markdown(f"```\n{content_preview}\n```")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Sidebar status
with st.sidebar:
    st.markdown("---")
    st.subheader("Status")
    
    # Check Weaviate connection
    try:
        client = connect_to_weaviate(weaviate_url)
        if client:
            if client.collections.exists(collection_name):
                st.success(f"✅ Connected to Weaviate")
                st.success(f"✅ Collection '{collection_name}' found")
            else:
                st.warning(f"⚠️ Collection '{collection_name}' not found")
            client.close()
        else:
            st.error("❌ Cannot connect to Weaviate")
    except:
        st.error("❌ Weaviate connection failed")
    
    # Check OpenRouter API
    if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        st.success("✅ OpenRouter API key configured")
        st.info(f"📋 Selected Model: {model_name}")
    else:
        st.error("❌ OpenRouter API key missing or invalid")
        st.info("💡 Get your API key from: https://openrouter.ai/")
        st.code("Add your key to the sidebar or update the .env file")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.processed = {}
        st.rerun()

if __name__ == "__main__":
    main()