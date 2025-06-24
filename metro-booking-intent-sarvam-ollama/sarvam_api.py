import streamlit as st
import requests
import json
import os

def transcribe_with_sarvam(audio_file_path, language_code, api_key, endpoint):
    """
    Sends an audio file to Sarvam AI for speech-to-text transcription.

    Args:
        audio_file_path (str): Path to the WAV audio file.
        language_code (str): Language code for transcription (e.g., "en-IN").
        api_key (str): Your Sarvam AI API subscription key.
        endpoint (str): Sarvam AI STT API endpoint URL.

    Returns:
        str: Transcribed text, or None if transcription fails.
    """
    if api_key is None:
        st.error("SARVAM_API_KEY not found. Please set it in your .env file.")
        return None
    try:
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
    except FileNotFoundError:
        st.error(f"Audio file not found: {audio_file_path}")
        return None

    headers = {"api-subscription-key": api_key}
    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {"language_code": language_code, "model": "saarika:v2"}
    st.info(f"Sending audio for transcription (language: {language_code})...")

    try:
        response = requests.post(endpoint, headers=headers, data=data, files=files)
        response.raise_for_status() # Raise HTTPError for bad responses
        result = response.json()
        if result and 'transcript' in result:
            st.success(f"Transcription successful for {language_code}:")
            st.code(result['transcript'])
            return result['transcript']
        else:
            st.warning(f"Transcription failed for {language_code}. Unexpected response format: {result}")
            return None
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh} - {response.text}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.error(f"Opaque Error: {err}")
    except Exception as e:
        st.error(f"An unexpected error occurred during STT: {e}")
    return None

def translate_text_with_sarvam(text_to_translate, source_language, target_language, api_key, endpoint):
    """
    Sends text to Sarvam AI for translation.

    Args:
        text_to_translate (str): The text to be translated.
        source_language (str): Source language code (e.g., "auto", "hi-IN").
        target_language (str): Target language code (e.g., "en-IN").
        api_key (str): Your Sarvam AI API subscription key.
        endpoint (str): Sarvam AI Translate API endpoint URL.

    Returns:
        str: Translated text, or None if translation fails.
    """
    if api_key is None:
        st.error("SARVAM_API_KEY not found. Please set it in your .env file.")
        return None
    if not text_to_translate:
        st.warning("No text to translate.")
        return None

    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    payload = {"input": text_to_translate, "source_language_code": source_language, "target_language_code": target_language}
    st.info(f"Sending text for translation (from {source_language} to {target_language})...")

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        if result and 'translated_text' in result:
            st.success(f"Translation successful ({source_language} to {target_language}):")
            st.code(result['translated_text'])
            return result['translated_text']
        else:
            st.warning(f"Translation failed ({source_language} to {target_language}). Unexpected response format: {result}")
            return None
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh} - {response.text}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.error(f"Opaque Error: {err}")
    except Exception as e:
        st.error(f"An unexpected error occurred during Translation: {e}")
    return None