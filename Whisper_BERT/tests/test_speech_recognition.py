#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Module for Speech Recognition

This module contains test cases for the speech recognition module.
"""

import os
import sys
import unittest
import tempfile
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.speech_recognition import SpeechRecognizer

class TestSpeechRecognition(unittest.TestCase):
    """Test cases for Speech Recognition Module"""
    
    def setUp(self):
        """Set up test environment"""
        # Use the smallest model for faster tests
        self.recognizer = SpeechRecognizer(model_size="tiny")
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_initialize_model(self):
        """Test model initialization"""
        self.assertIsNotNone(self.recognizer.model)
        self.assertIsInstance(self.recognizer.model, object)
    
    def test_multiple_language_models(self):
        """Test model initialization for different languages"""
        languages = ['en', 'hi']
        for lang in languages:
            recognizer = SpeechRecognizer(language=lang, model_size="tiny")
            self.assertIsNotNone(recognizer.model)
    
    def create_dummy_audio(self, duration=3, sample_rate=16000):
        """Create a dummy audio file for testing"""
        # Create a dummy audio file (sine wave)
        audio_path = os.path.join(self.test_dir, "test_audio.wav")
        
        # Create a simple sine wave
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        
        # Save as WAV file
        import torchaudio
        torchaudio.save(audio_path, waveform, sample_rate)
        
        return audio_path

if __name__ == '__main__':
    unittest.main()
