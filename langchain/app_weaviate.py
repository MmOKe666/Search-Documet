import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import weaviate
from weaviate.classes.query import MetadataQuery
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chat with DeepSeek & Weaviate",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chat Application")
st.markdown("*Powered by DeepSeek LLM and Weaviate Vector Database*")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # DeepSeek API configuration
    deepseek_api_key = st.text_input(
        "DeepSeek API Key", 
        type="password", 
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        help="Enter your DeepSeek API key"
    )
    
    # Weaviate configuration
    weaviate_url = st.text_input(
        "Weaviate URL", 
        value="http://localhost:8080",
        help="Weaviate instance URL"
    )
    
    collection_name = st.text_input(
        "Collection Name", 
        value="Document",
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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Initialize DeepSeek LLM
@st.cache_resource
def load_deepseek_llm(api_key):
    """Load and cache the DeepSeek LLM"""
    if not api_key:
        st.error("Please provide DeepSeek API key in the sidebar")
        return None
    
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
        max_tokens=2000
    )

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
        
        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["title", "content", "metadata"]
        )
        
        # Convert results to LangChain documents
        documents = []
        for obj in response.objects:
            doc_content = obj.properties.get("content", "")
            doc_title = obj.properties.get("title", "Unknown")
            doc_metadata = obj.properties.get("metadata", {})
            
            # Add distance to metadata
            if hasattr(obj.metadata, 'distance'):
                doc_metadata["distance"] = obj.metadata.distance
            
            documents.append(Document(
                page_content=doc_content,
                metadata={
                    "title": doc_title,
                    "source": doc_metadata.get("source", "Unknown"),
                    "distance": doc_metadata.get("distance", 0)
                }
            ))
        
        return documents
        
    except Exception as e:
        st.error(f"Error querying Weaviate: {e}")
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

# Create RAG chain
def create_rag_chain(llm):
    """Create the RAG chain for question answering"""
    system_prompt = (
        "You are a helpful AI assistant that answers questions based on the provided context. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Be concise but comprehensive in your response. "
        "Always cite the source when possible.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return create_stuff_documents_chain(llm, prompt)

# Custom retriever class for Weaviate
class WeaviateRetriever:
    def __init__(self, weaviate_url, collection_name, limit=3):
        self.weaviate_url = weaviate_url
        self.collection_name = collection_name
        self.limit = limit
    
    def get_relevant_documents(self, query):
        return query_weaviate(query, self.weaviate_url, self.collection_name, self.limit)
    
    def invoke(self, input_dict):
        query = input_dict.get("input", "")
        return self.get_relevant_documents(query)

# Main chat interface
def main():
    # Load models
    embedding_model = load_embedding_model()
    llm = load_deepseek_llm(deepseek_api_key)
    
    if not llm:
        st.warning("Please configure DeepSeek API key to start chatting")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['title']}")
                        st.markdown(f"*Distance: {source.get('distance', 'N/A'):.4f}*")
                        st.markdown(f"```\n{source['content'][:200]}...\n```")
    
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
                    
                    if not relevant_docs:
                        response = "I couldn't find any relevant documents to answer your question. Please make sure your Weaviate database contains documents and is properly configured."
                        sources = []
                    else:
                        # Create RAG chain
                        qa_chain = create_rag_chain(llm)
                        
                        # Generate response
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        response = qa_chain.invoke({
                            "context": context,
                            "input": prompt
                        })
                        
                        # Prepare sources for display
                        sources = [
                            {
                                "title": doc.metadata.get("title", "Unknown"),
                                "content": doc.page_content,
                                "distance": doc.metadata.get("distance", 0)
                            }
                            for doc in relevant_docs
                        ]
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source['title']}")
                                st.markdown(f"*Distance: {source.get('distance', 'N/A'):.4f}*")
                                st.markdown(f"```\n{source['content'][:200]}...\n```")
                    
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
    
    # Check DeepSeek API
    if deepseek_api_key:
        st.success("✅ DeepSeek API key configured")
    else:
        st.error("❌ DeepSeek API key missing")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()