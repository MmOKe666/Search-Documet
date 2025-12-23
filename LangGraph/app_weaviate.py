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
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from typing import Any


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # You can set the level to INFO, WARNING, etc., as needed.
# Create a logger
logger = logging.getLogger(__name__)

# Import streamlit (missing import statement)
import streamlit as st

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    context: str
    answer: str
    sources: List[dict]
    llm: Any
    search_alpha: float

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

openrouter_api_key = None
yandex_api_key = None
yandex_folder_id = None
# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["Yandex Cloud", "OpenRouter"],
        index=0
    )

    if llm_provider == "Yandex Cloud":
        yandex_api_key = st.text_input(
            "Yandex Cloud API Key",
            type="password",
            value=os.getenv("YANDEX_API_KEY", "")
        )

        yandex_folder_id = st.text_input(
            "Yandex Cloud Folder ID",
            value=os.getenv("YANDEX_FOLDER_ID", "")
        )

        model_name = st.selectbox(
            "Yandex Model",
            ["yandexgpt", "yandexgpt-lite", "summarize"]
        )

    elif llm_provider == "OpenRouter":
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=os.getenv("OPENROUTER_API_KEY", "")
        )

        model_name = st.text_input(
            "OpenRouter Model",
            value=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
        )

    search_mode = st.selectbox(
        "Search mode",
        ["Hybrid", "Vector", "BM25"],
        index=0
    )

    if search_mode == "Vector":
        alpha = 0.0
    elif search_mode == "BM25":
        alpha = 1.0
    else:
        alpha = 0.5

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

def load_llm(provider: str, model: str, **kwargs):
    try:
        if provider == "Yandex Cloud":
            from langchain_community.llms import YandexGPT

            api_key = kwargs.get("api_key")
            folder_id = kwargs.get("folder_id")

            if not api_key or not folder_id:
                st.error("❌ Yandex Cloud API key or Folder ID is missing")
                return None

            # YandexGPT читает ТОЛЬКО environment variables
            os.environ["YC_API_KEY"] = api_key
            os.environ["YC_FOLDER_ID"] = folder_id

            return YandexGPT(
                model_name=model,
                temperature=0.1,
                max_tokens=2000
            )

        elif provider == "OpenRouter":
            api_key = kwargs.get("api_key")

            if not api_key:
                st.error("❌ OpenRouter API key is missing")
                return None

            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=2000
            )

    except Exception as e:
        st.error(f"Error initializing {provider} LLM: {e}")
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


def query_weaviate(
    query_text: str,
    weaviate_url: str,
    collection_name: str,
    limit: int = 3,
    alpha: float = 0.5
):
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

        # Semantic search
        # response = collection.query.near_vector(
        #     near_vector=query_vector,
        #     limit=limit,
        #     return_metadata=MetadataQuery(distance=True)
        #     # Removed return_properties to get all available properties
        # )

        # Hybrid search
        response = collection.query.hybrid(
            query=query_text,  # request text
            vector=query_vector,  # query vector (for semantic components)
            alpha=alpha,  # balance: 0.5 = mixed search, 0 - sematic, 1 - BM25
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
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

MAX_CONTEXT_CHARS = 6000

def retrieve_docs(state: RAGState):
    docs = query_weaviate(
        state["question"],
        weaviate_url,
        collection_name,
        search_limit,
        alpha = state["search_alpha"]
    )
    return {"documents": docs}


def build_context(state: RAGState):
    context_parts = []
    sources = []

    for doc in state["documents"]:
        distance = doc.metadata.get("distance", 1.0)

        if distance <= 0.8:
            content = str(doc.page_content)
            context_parts.append(content)

            sources.append({
                "title": str(doc.metadata.get("title", "Unknown")),
                "content": content,
                "distance": float(distance)
            })

    context = "\n\n".join(context_parts)
    context = context[:MAX_CONTEXT_CHARS]

    if not context:
        context = "No relevant content found."

    return {
        "context": context,
        "sources": sources
    }


def generate_answer(state: RAGState):
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the question based on the provided context. "
        "If context does not contain the requested information, say "
        "'I don't have enough information to answer this question.'\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    chain = prompt | state["llm"]
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("context", build_context)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "context")
    graph.add_edge("context", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

# Main chat interface
def main():
    # Load models
    embedding_model = load_embedding_model()
    if llm_provider == "Yandex Cloud":
        llm = load_llm(
            provider="Yandex Cloud",
            model=model_name,
            api_key=yandex_api_key,
            folder_id=yandex_folder_id
        )
    else:
        llm = load_llm(
            provider="OpenRouter",
            model=model_name,
            api_key=openrouter_api_key
        )

    if not llm:
        st.warning(f"⚠️ Please configure valid {llm_provider} credentials to start chatting")
        st.info("📝 Steps to get started:")
        st.markdown("""
           1. Go to [Yandex Cloud](https://cloud.yandex.com/) and create an account
           2. Create a folder in Yandex Cloud
           3. Enable the YandexGPT API service
           4. Generate an API key and get Folder ID
           5. Enter the credentials in the sidebar
           """)
        return

    rag_app = build_rag_graph()

    # Display chat history
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['title']}")
                        st.markdown(f"*Distance: {source.get('distance', 'N/A'):.4f}*")

                        # Используем чекбокс для показа полного контента
                        show_full = st.checkbox(f"Show full content of Source {i}",
                                                key=f"hist_check_{message_index}_{i}")

                        if show_full:
                            st.text_area("",
                                         value=source['content'],
                                         height=200,
                                         key=f"hist_content_{message_index}_{i}")
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
                    result = rag_app.invoke({
                        "question": prompt,
                        "llm": llm,
                        "search_alpha": alpha
                    })

                    response = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")


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

    # Check LLM API
    if llm_provider == "Yandex Cloud":
        if yandex_api_key and yandex_folder_id:
            st.success("✅ Yandex Cloud API key configured")
            st.success("✅ Yandex Cloud Folder ID configured")
            st.info(f"📋 Selected Model: {model_name}")
        else:
            if not yandex_api_key:
                st.error("❌ Yandex Cloud API key missing")
            if not yandex_folder_id:
                st.error("❌ Yandex Cloud Folder ID missing")

    elif llm_provider == "OpenRouter":
        if openrouter_api_key:
            st.success("✅ OpenRouter API key configured")
            st.info(f"📋 Selected Model: {model_name}")
        else:
            st.error("❌ OpenRouter API key missing")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.processed = {}

        for key in list(st.session_state.keys()):
            if key.startswith(('hist_check_', 'curr_check_')):
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()