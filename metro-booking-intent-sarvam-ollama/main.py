import streamlit as st
import os
import tempfile


from config import (
    SARVAM_API_KEY, OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME,
    SARVAM_STT_ENDPOINT, SARVAM_TRANSLATE_ENDPOINT,
    SAMPLE_RATE, CHANNELS, DURATION_SECONDS, RAG_KNOWLEDGE_BASE,
    SARVAM_STT_LANGUAGES, SARVAM_TRANSLATE_LANGUAGES
)
from audio_processing import record_audio_to_file
from sarvam_api import transcribe_with_sarvam, translate_text_with_sarvam
from rag_system import load_rag_components, get_llm_intent_rag
from llm_interface import call_ollama_chat_api

# Main Streamlit UI
st.set_page_config(page_title="Sarvam AI + RAG LLM Metro Booking Demo", layout="centered")

# Initialize RAG Components---
embedding_model, rag_document_embeddings = load_rag_components()

# Initialize Session State

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "translated_text_for_llm" not in st.session_state:
    st.session_state.translated_text_for_llm = ""
if "audio_file_path" not in st.session_state:
    st.session_state.audio_file_path = None

st.title("Sarvam AI STT & Translation + RAG LLM for Metro Booking")
st.markdown("""
This application demonstrates:
1.  **Speech-to-Text (STT)** using Sarvam AI.
2.  **Text Translation** using Sarvam AI.
3.  **Intent Recognition** for Metro Booking using a local Open-Source LLM (Ollama) augmented with **Retrieval Augmented Generation (RAG)**.
""")

# STT Section
st.header("1. Speech-to-Text (STT)")
st.markdown("Record your voice and get it transcribed.")

st.sidebar.header("STT Settings")
stt_language = st.sidebar.selectbox(
    "Select STT Language:",
    options=SARVAM_STT_LANGUAGES,
    index=0,
    key="stt_lang_select"
)
stt_duration = st.sidebar.slider(
    "Recording Duration (seconds):",
    min_value=1,
    max_value=15,
    value=DURATION_SECONDS,
    step=1,
    key="stt_duration_slider"
)

st.session_state.transcribed_text = st.session_state.get('transcribed_text', "")

if st.button(f"Start Recording for {stt_duration} seconds ({stt_language})", key="record_button"):
    st.session_state.audio_file_path = record_audio_to_file(
        duration=stt_duration,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS
    )
    if st.session_state.audio_file_path:
        st.audio(st.session_state.audio_file_path, format='audio/wav')
        with st.spinner("Transcribing audio..."):
            st.session_state.transcribed_text = transcribe_with_sarvam(
                st.session_state.audio_file_path, stt_language, SARVAM_API_KEY, SARVAM_STT_ENDPOINT
            )
    else:
        st.session_state.transcribed_text = ""

if st.session_state.transcribed_text:
    st.subheader("Transcribed Text:")
    st.text_area("Your Transcribed Text", value=st.session_state.transcribed_text, height=100, key="stt_output_text")

# --- Translation Section ---
st.header("2. Text Translation (Optional)")
st.markdown("Translate the transcribed text or manually entered text. **Translate to English (en-IN) for best LLM performance.**")

translation_input_text = st.text_area(
    "Text to Translate (or derived from STT above):",
    value=st.session_state.transcribed_text if st.session_state.transcribed_text else "",
    height=100,
    key="translation_input"
)

st.sidebar.header("Translation Settings")
source_lang_trans = st.sidebar.selectbox(
    "Source Language for Translation:",
    options=SARVAM_TRANSLATE_LANGUAGES,
    index=0,
    key="source_lang_trans_select"
)
# Force target to English for LLM processing
target_lang_trans_for_llm = "en-IN" # Hardcoded for LLM compatibility

st.session_state.translated_text_for_llm = st.session_state.get('translated_text_for_llm', "")

if st.button(f"Translate Text to {target_lang_trans_for_llm} (for LLM)", key="translate_for_llm_button"):
    if translation_input_text:
        with st.spinner("Translating text..."):
            st.session_state.translated_text_for_llm = translate_text_with_sarvam(
                translation_input_text, source_lang_trans, target_lang_trans_for_llm,
                SARVAM_API_KEY, SARVAM_TRANSLATE_ENDPOINT
            )
    else:
        st.warning("Please provide text to translate.")

if st.session_state.translated_text_for_llm:
    st.subheader(f"Translated Text ({target_lang_trans_for_llm} for LLM Processing):")
    st.text_area("Translated", value=st.session_state.translated_text_for_llm, height=100, disabled=True, key="translated_output_text")
    st.info("This text will be sent to the LLM for intent recognition with RAG.")
else:
    st.info("Translate text above to prepare for LLM intent recognition.")

# --- LLM Intent Recognition Section with RAG ---
st.header("3. LLM Intent Recognition & Booking (with RAG)")
st.markdown("""
The translated text will be sent to a local Open-Source LLM (powered by Ollama),
which will use RAG to find relevant intent examples from its knowledge base.
If the intent is 'Book Metro Ticket', a mock booking will be confirmed.
""")

st.info(f"**Ollama Configuration:** Base URL: `{OLLAMA_API_BASE_URL}`, Model: `{OLLAMA_MODEL_NAME}`")
st.warning("Make sure Ollama is running and you have pulled the specified model (e.g., `ollama run llama3`) before proceeding.")
st.info(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`. This model will be downloaded the first time you run.")

if st.button("Identify Intent with RAG & Book (if applicable)", key="identify_intent_rag_button"):
    if not st.session_state.translated_text_for_llm:
        st.warning("Please ensure you have translated text available for intent recognition.")
    elif not embedding_model or rag_document_embeddings is None:
        st.error("RAG system is not ready. Please check embedding model loading.")
    else:
        with st.spinner("Identifying intent with LLM and RAG..."):
            llm_intent = get_llm_intent_rag(
                user_input_text=st.session_state.translated_text_for_llm,
                embedding_model=embedding_model,
                rag_document_embeddings=rag_document_embeddings,
                rag_documents_list=RAG_KNOWLEDGE_BASE, # Pass the original KNOWLEDGE_BASE for displaying
                ollama_model_name=OLLAMA_MODEL_NAME,
                ollama_api_base_url=OLLAMA_API_BASE_URL,
                top_k=2 # You can make this configurable in config.py if needed
            )

            if llm_intent == "INTENT_BOOK_TICKET":
                st.success("**Intent Recognized: Book Metro Ticket!**")
                st.balloons()
                st.info(f"Simulating booking for: '{st.session_state.translated_text_for_llm}'")
                st.success("**Metro Ticket Booked Successfully!** (This is a simulated booking)")
                st.markdown("You can enhance this section to extract details like 'from' and 'to' stations by refining the RAG prompt and LLM parsing.")
            elif llm_intent == "INTENT_CHECK_BALANCE":
                st.info("**Intent Recognized: Check Metro Balance.**")
                st.write("Your metro card balance is ₹150. (This is a simulated response)")
            elif llm_intent == "INTENT_CANCEL_BOOKING":
                st.info("**Intent Recognized: Cancel Booking.**")
                st.write("Your last metro booking has been cancelled. (This is a simulated response)")
            elif llm_intent == "INTENT_OTHER":
                st.info("**Intent Recognized: Other.**")
                st.warning("The LLM did not identify a specific metro booking intent.")
                st.write("Please try rephrasing your request.")
            else:
                st.error(f"An error occurred during LLM intent recognition: {llm_intent}")

st.markdown("---")
st.caption("Powered by Sarvam AI, Ollama, and RAG")
st.caption("Developed by Your Name/Organization")



# Cleanup temporary audio file
if 'audio_file_path' in st.session_state and st.session_state.audio_file_path is not None:
    if os.path.exists(st.session_state.audio_file_path):
        try:
            os.remove(st.session_state.audio_file_path)
            del st.session_state['audio_file_path']
        except Exception as e:
            st.warning(f"Could not remove temporary audio file: {e}")
    else:
        del st.session_state['audio_file_path']