#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Speech Recognition Module

This module is responsible for transcribing speech input to text using 
language-specific ASR models. It supports multiple Indian languages:
English, Hindi, Tamil, Telugu, Kannada, and Malayalam.
"""

import os
import torch
import torchaudio
import whisper
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import ssl
import platform
import sys

class SpeechRecognizer:
    """
    Class for automatic speech recognition in multiple Indian languages.
    """
    
    def __init__(self, language='en', model_type='whisper', model_size='medium', device=None):
        """
        Initialize the speech recognition model.
        
        Args:
            language (str): Language code for transcription
            model_type (str): Type of model to use ('whisper', 'indic-asr')
            model_size (str): Size of model ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.language = language
        self.model_type = model_type
        
        # Model mapping for different languages and model types
        self.model_map = {
            'whisper': {
                'en': f"openai/whisper-{model_size}.en",
                'hi': f"openai/whisper-{model_size}",
                'ta': f"openai/whisper-{model_size}",
                'te': f"openai/whisper-{model_size}",
                'kn': f"openai/whisper-{model_size}",
                'ml': f"openai/whisper-{model_size}",
            },
            'indic-asr': {
                'en': "ai4bharat/conformer-en-hi",
                'hi': "ai4bharat/conformer-hi",
                'ta': "ai4bharat/conformer-ta",
                'te': "ai4bharat/conformer-te",
                'kn': "ai4bharat/conformer-kn",
                'ml': "ai4bharat/conformer-ml",
            }
        }
        
        # Load the appropriate model based on language and model type
        print(f"Initializing speech recognition model for {language} using {model_type} on {self.device}")
        
        if model_type == 'whisper':
            self._initialize_whisper_model(model_size)
        elif model_type == 'indic-asr':
            self._initialize_indic_asr_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _initialize_whisper_model(self, model_size):
        """Initialize Whisper model for speech recognition."""
        try:
            if self.language == 'en':
                # Use English-specific model for better accuracy
                self.model = whisper.load_model(f"{model_size}.en")
            else:
                # Use multilingual model for other languages
                self.model = whisper.load_model(model_size)
            
            self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    def _initialize_indic_asr_model(self):
        """Initialize Indic ASR model from AI4Bharat for speech recognition."""
        try:
            model_id = self.model_map['indic-asr'].get(self.language, "ai4bharat/conformer-hi")
            
            # Using HuggingFace Transformers pipeline for Indic ASR
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading Indic ASR model: {e}")
            print("Falling back to Whisper model")
            self._initialize_whisper_model("medium")
    
    def transcribe(self, audio_path):
        """
        Transcribe speech to text.
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            str: Transcribed text
        """
        try:
            if self.model_type == 'whisper':
                return self._transcribe_with_whisper(audio_path)
            elif self.model_type == 'indic-asr':
                return self._transcribe_with_indic_asr(audio_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
    
    def _transcribe_with_whisper(self, audio_path):
        """Transcribe using Whisper model."""
        # Set language-specific options
        options = {
            "language": self.language,
            "fp16": self.device == "cuda"
        }
        
        # Perform transcription
        result = self.model.transcribe(audio_path, **options)
        return result["text"].strip()
    
    def _transcribe_with_indic_asr(self, audio_path):
        """Transcribe using Indic ASR model."""
        result = self.asr_pipeline(audio_path)
        return result["text"].strip()

# Testing function
def test_speech_recognition(audio_path, language='en'):
    """Test the speech recognition module with a sample audio."""
    recognizer = SpeechRecognizer(language=language)
    transcribed_text = recognizer.transcribe(audio_path)
    
    print(f"Language: {language}")
    print(f"Transcribed text: {transcribed_text}")
    
    return transcribed_text

if __name__ == "__main__":
    # Test with a sample audio if available
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else 'en'
        
        if os.path.exists(audio_path):
            test_speech_recognition(audio_path, language)
        else:
            print(f"Audio file {audio_path} not found.")
    else:
        print("Please provide an audio file path as argument.")
