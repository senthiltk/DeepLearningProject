import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import tempfile
import os

def record_audio_to_file(duration, sample_rate, channels):
    """
    Records audio from the microphone and saves it to a temporary WAV file.

    Args:
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sample rate for audio recording (e.g., 16000).
        channels (int): Number of audio channels (e.g., 1 for mono).

    Returns:
        str: Path to the temporary WAV file, or None if an error occurs.
    """
    st.info(f"Recording for {duration} seconds. Please speak clearly...")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()
        st.success("Recording complete!")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            wav_file_path = tmp_file.name
            wavio.write(wav_file_path, audio_data, rate=sample_rate, sampwidth=2)
        return wav_file_path
    except Exception as e:
        st.error(f"Error during audio recording: {e}")
        st.warning("Please ensure your microphone is connected and working. You might need to install PortAudio drivers or check permissions.")
        return None