#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Language Identification Module

This module is responsible for identifying the language of a speech input.
It uses pre-trained models to determine the language among the supported
languages: English, Hindi, Tamil, Telugu, Kannada, and Malayalam.
"""

import os
import torch
import torchaudio
import whisper
import numpy as np
from pathlib import Path
import ssl
import tempfile
import warnings

class LanguageIdentifier:
    """
    Class to identify the language of spoken input using pre-trained models.
    """
    
    def __init__(self, model_name="base", device=None):
        """
        Initialize the language identification model.
        
        Args:
            model_name (str): Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        # Fix SSL certificate verification issues (especially on macOS)
        ssl._create_default_https_context = ssl._create_unverified_context
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading language identification model: {model_name} on {self.device}")
        try:
            # Try different approaches to load the model
            try:
                self.model = whisper.load_model(model_name)
            except Exception as e:
                print(f"Standard model loading failed: {e}, trying with explicit download cache...")
                # Use a more reliable cache location
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
                os.makedirs(cache_dir, exist_ok=True)
                self.model = whisper.load_model(model_name, download_root=cache_dir)
                
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading whisper model: {e}")
            print("Falling back to tiny model which should be more reliable to download...")
            try:
                self.model = whisper.load_model("tiny")
                self.model.to(self.device)
            except Exception as e2:
                print(f"Critical error: Failed to load even the tiny model: {e2}")
                raise
            
        # Define mapping of Whisper language codes to full language names
        self.language_map = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam'
        }
        
        # Supported languages for this project
        self.supported_languages = set(self.language_map.keys())
        
    def identify(self, audio_path):
        """
        Identify the language of the given audio.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Language code of the identified language
        """
        try:
            # Load audio using whisper's utility for compatibility
            audio = whisper.load_audio(audio_path)
            
            # Detect language
            # Use only first 30 seconds for detection to improve speed
            audio = audio[:min(len(audio), 30 * 16000)]
            
            # Use Whisper's detect_language method
            audio_tensor = torch.from_numpy(audio).to(self.device)
            mel = whisper.log_mel_spectrogram(audio_tensor)
            
            # Get language probability distribution
            _, probs = self.model.detect_language(mel)
            
            # Get the most likely language
            detected_lang = max(probs, key=probs.get)
            
            # Filter to supported languages if needed
            if detected_lang not in self.supported_languages:
                print(f"Warning: Detected language '{detected_lang}' is not in supported languages.")
                
                # Find the highest probability supported language
                supported_probs = {lang: probs[lang] for lang in self.supported_languages if lang in probs}
                if supported_probs:
                    detected_lang = max(supported_probs, key=supported_probs.get)
                    print(f"Using closest supported language: {detected_lang}")
                else:
                    print("No supported language detected. Defaulting to English.")
                    detected_lang = 'en'
            
            return detected_lang
            
        except Exception as e:
            print(f"Error during language identification: {e}")
            print("Defaulting to English")
            return 'en'
            
    def get_language_name(self, language_code):
        """
        Get the full language name from the language code.
        
        Args:
            language_code (str): The language code
            
        Returns:
            str: The full language name
        """
        return self.language_map.get(language_code, "Unknown")
        
# Testing function
def test_language_identification(audio_path):
    """Test the language identification module with a sample audio."""
    identifier = LanguageIdentifier()
    detected_language = identifier.identify(audio_path)
    language_name = identifier.get_language_name(detected_language)
    
    print(f"Detected language code: {detected_language}")
    print(f"Detected language name: {language_name}")
    
    return detected_language

if __name__ == "__main__":
    # Test with a sample audio if available
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if os.path.exists(audio_path):
            test_language_identification(audio_path)
        else:
            print(f"Audio file {audio_path} not found.")
    else:
        print("Please provide an audio file path as argument.")
