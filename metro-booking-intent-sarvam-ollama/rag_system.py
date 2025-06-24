import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

from config import EMBEDDING_MODEL_NAME, RAG_KNOWLEDGE_BASE
from llm_interface import call_ollama_chat_api # Import the LLM calling function

# Prepare RAG documents
def _prepare_rag_documents(knowledge_base):
    rag_documents = []
    for item in knowledge_base:
        combined_text = f"Intent: {item['intent_name']}. Description: {item['description']} Examples: " + " ".join(item['examples'])
        rag_documents.append({
            "text": combined_text,
            "intent_name": item['intent_name']
        })
    return rag_documents

RAG_DOCUMENTS = _prepare_rag_documents(RAG_KNOWLEDGE_BASE)

@st.cache_resource
def load_embedding_model(model_name):
    """
    Loads the SentenceTransformer embedding model.
    Cached to load only once.
    """
    with st.spinner(f"Loading embedding model: {model_name}..."):
        try:
            model = SentenceTransformer(model_name)
            st.success("Embedding model loaded!")
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            st.info("Please ensure 'sentence-transformers' is installed: `pip install sentence-transformers`")
            return None

@st.cache_data(show_spinner=False)
def get_document_embeddings(documents, _model):
    """
    Computes embeddings for the RAG documents.
    Cached to compute only once.
    """
    if not _model: return None
    texts = [doc["text"] for doc in documents]
    st.info("Computing embeddings for RAG knowledge base...")
    embeddings = _model.encode(texts, convert_to_tensor=False)
    st.success("RAG knowledge base embedded!")
    return embeddings

def load_rag_components():
    """
    Loads the embedding model and computes/retrieves document embeddings.
    """
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if embedding_model:
        rag_document_embeddings = get_document_embeddings(RAG_DOCUMENTS, embedding_model)
    else:
        rag_document_embeddings = None
    return embedding_model, rag_document_embeddings

def get_llm_intent_rag(user_input_text, embedding_model, rag_document_embeddings, rag_documents_list, ollama_model_name, ollama_api_base_url, top_k=2):
    """
    Sends text to a local Ollama LLM for intent recognition using RAG.
    Retrieves relevant intent examples from the RAG_DOCUMENTS to augment the prompt.
    """
    if not user_input_text:
        return "No input text for LLM."
    if not embedding_model or rag_document_embeddings is None:
        return "RAG system not initialized (embedding model or documents missing)."

    st.info(f"Using RAG to find relevant intent examples for: '{user_input_text}'")

    # 1. Embed the user query
    user_query_embedding = embedding_model.encode([user_input_text], convert_to_tensor=False)

    # 2. Retrieve top_k most similar documents
    similarities = cosine_similarity(user_query_embedding, rag_document_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1] # Get indices of top_k highest similarities

    retrieved_context = []
    st.subheader("Retrieved Context from RAG:")
    for i, idx in enumerate(top_indices):
        # Use RAG_KNOWLEDGE_BASE for display as it has original intent names and descriptions
        # RAG_DOCUMENTS is for embedding, combined text might be too long for display
        doc = RAG_KNOWLEDGE_BASE[idx]
        similarity_score = similarities[idx]
        st.write(f"- **Intent:** {doc['intent_name']} (Similarity: {similarity_score:.2f})")
        # Displaying a snippet of the combined text used for RAG
        combined_text_snippet = f"Description: {doc['description']} Examples: " + " ".join(doc['examples'])
        st.write(f"  Snippet: {combined_text_snippet[:150]}...")
        retrieved_context.append(RAG_DOCUMENTS[idx]["text"]) # Add the full combined text for LLM prompt

    context_str = "\n".join(retrieved_context)

    # 3. Construct the prompt with retrieved context
    prompt = f"""You are an intelligent assistant for a metro booking application.
    Your task is to identify the user's intent based on their query and the provided relevant examples.

    Here is some relevant information and examples of different intents from our knowledge base:
    <context>
    {context_str}
    </context>

    Based on the user's input and the context above, determine the primary intent.
    If the intent is to 'Book Metro Ticket', respond with 'INTENT_BOOK_TICKET'.
    If the intent is to 'Check Metro Balance', respond with 'INTENT_CHECK_BALANCE'.
    If the intent is to 'Cancel Booking', respond with 'INTENT_CANCEL_BOOKING'.
    If the intent is any other general query, respond with 'INTENT_OTHER'.

    User Input: "{user_input_text}"

    Your response (e.g., INTENT_BOOK_TICKET, INTENT_CHECK_BALANCE, INTENT_CANCEL_BOOKING, INTENT_OTHER):
    """

    messages = [{"role": "user", "content": prompt}]

    llm_response_text = call_ollama_chat_api(ollama_api_base_url, ollama_model_name, messages)

    if llm_response_text:
        st.success(f"LLM Response: {llm_response_text}")
        if "INTENT_BOOK_TICKET" in llm_response_text:
            return "INTENT_BOOK_TICKET"
        elif "INTENT_CHECK_BALANCE" in llm_response_text:
            return "INTENT_CHECK_BALANCE"
        elif "INTENT_CANCEL_BOOKING" in llm_response_text:
            return "INTENT_CANCEL_BOOKING"
        else:
            return "INTENT_OTHER"
    else:
        st.error("Failed to get a response from the LLM.")
        return "ERROR_LLM_RESPONSE"