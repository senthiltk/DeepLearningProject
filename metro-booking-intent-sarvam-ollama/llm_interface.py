import streamlit as st
import requests
import json

def call_ollama_chat_api(base_url, model_name, messages, timeout=180):
    """
    Makes a call to the Ollama chat API.

    Args:
        base_url (str): The base URL of the Ollama server (e.g., "http://localhost:11434").
        model_name (str): The name of the Ollama model to use (e.g., "llama3").
        messages (list): A list of message dictionaries for the chat API.
        timeout (int): Timeout for the API request in seconds.

    Returns:
        str: The content of the LLM's response, or None if an error occurs.
    """
    try:
        st.info(f"Asking LLM ({model_name}) to identify intent with RAG context...")
        response = requests.post(
            f"{base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            json={"model": model_name, "messages": messages, "stream": False},
            timeout=timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result and 'message' in result and 'content' in result['message']:
            return result['message']['content'].strip()
        else:
            st.warning(f"LLM response format unexpected: {result}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to Ollama server at {base_url}. Is Ollama running?")
        st.info("Please ensure Ollama is installed and running, and the model is pulled (`ollama run {model_name}`).")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred with LLM: {e}")
        return None