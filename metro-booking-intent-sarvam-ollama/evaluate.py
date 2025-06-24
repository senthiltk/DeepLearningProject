# evaluate.py
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np

# Import necessary functions/configurations from your project
from config import (
    OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME, RAG_KNOWLEDGE_BASE
)
from rag_system import load_rag_components, get_llm_intent_rag
from llm_interface import call_ollama_chat_api # Needed indirectly by rag_system

# --- Disable Streamlit's st.info/st.warning during evaluation ---
# This is a simple way to prevent Streamlit components from running
# when you're running the script outside the Streamlit environment.
# For a more robust solution, you might refactor your functions to
# accept a 'logger' argument instead of directly calling st.
def noop(*args, **kwargs):
    pass

import streamlit as st
st.info = noop
st.warning = noop
st.error = noop
st.success = noop
st.code = noop
st.write = noop
st.subheader = noop
st.markdown = noop
st.text_area = noop
st.header = noop
st.button = noop # If you had buttons in the evaluation path
st.spinner = lambda x: type('obj', (object,), {'__enter__': noop, '__exit__': noop})()
# --- End disable Streamlit calls ---


def load_test_data(filepath="test_data.csv"):
    """Loads test data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def run_evaluation():
    print("Loading RAG components...")
    embedding_model, rag_document_embeddings = load_rag_components()

    if embedding_model is None or rag_document_embeddings is None:
        print("Error: RAG components failed to load. Exiting evaluation.")
        return

    test_data = load_test_data()
    true_labels = []
    predicted_labels = []

    print(f"\nStarting evaluation on {len(test_data)} samples...")
    for index, row in test_data.iterrows():
        user_query = row["User Query"]
        true_intent = row["True Intent"]

        # Simulate the translation step for the LLM
        # In a real scenario, if your test data is in a non-English language,
        # you would call translate_text_with_sarvam here.
        # For simplicity, assuming test data is already in English for LLM.
        processed_query_for_llm = user_query # Assuming English test data

        print(f"Processing '{user_query}' (True: {true_intent})...")
        predicted_intent_raw = get_llm_intent_rag(
            user_input_text=processed_query_for_llm,
            embedding_model=embedding_model,
            rag_document_embeddings=rag_document_embeddings,
            rag_documents_list=RAG_KNOWLEDGE_BASE,
            ollama_model_name=OLLAMA_MODEL_NAME,
            ollama_api_base_url=OLLAMA_API_BASE_URL,
            top_k=2
        )

        # Map the LLM's raw output to your target intent names
        # You need to adjust this logic based on what get_llm_intent_rag actually returns
        # For example, if it returns "INTENT_BOOK_TICKET", map it to "Book Metro Ticket"
        predicted_intent_map = {
            "INTENT_BOOK_TICKET": "Book Metro Ticket",
            "INTENT_CHECK_BALANCE": "Check Metro Balance",
            "INTENT_CANCEL_BOOKING": "Cancel Booking",
            "INTENT_OTHER": "General Query", # Assuming "General Query" is your label for "Other"
            "ERROR_LLM_RESPONSE": "ERROR" # Handle errors
        }
        # Get the actual intent name from the predicted raw string
        final_predicted_intent = "ERROR" # Default to error
        for key, value in predicted_intent_map.items():
            if key in predicted_intent_raw:
                final_predicted_intent = value
                break

        true_labels.append(true_intent)
        predicted_labels.append(final_predicted_intent)
        print(f"  -> Predicted: {final_predicted_intent}\n")

    # --- Metrics Calculation ---
    print("\n--- Evaluation Results ---")

    # Define all possible classes (intents)
    # Ensure this list covers all intents in your RAG_KNOWLEDGE_BASE and test_data
    # and also includes 'ERROR' if you predict it.
    classes = sorted(list(set(true_labels + predicted_labels)))
    print(f"Classes: {classes}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    print("\nConfusion Matrix:")
    # For better readability, you might use a library like seaborn or matplotlib
    # to plot this. Here's a basic print:
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    print(df_cm)

    # F1 Score and Classification Report
    # 'weighted' F1 score is good for imbalanced datasets as it accounts for class support
    f1 = f1_score(true_labels, predicted_labels, average='weighted', labels=classes, zero_division=0)
    print(f"\nWeighted F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    report = classification_report(true_labels, predicted_labels, labels=classes, zero_division=0)
    print(report)

if __name__ == "__main__":
    run_evaluation()