#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated Test Module for Metro Booking Voice Assistant

This module contains test cases for the integrated functionality
of the Metro Booking Voice Assistant.
"""

import os
import sys
import unittest
import tempfile
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import process_voice_message
from modules.language_identification import LanguageIdentifier
from modules.speech_recognition import SpeechRecognizer
from modules.intent_detection import IntentDetector

class TestIntegration(unittest.TestCase):
    """Test cases for integrated functionality of the Metro Booking Voice Assistant"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test audio files
        self.english_audio = self.create_dummy_audio('en')
        self.hindi_audio = self.create_dummy_audio('hi')
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.test_dir)
    
    def create_dummy_audio(self, language, duration=3, sample_rate=16000):
        """Create a dummy audio file for testing"""
        # Create a dummy audio file (sine wave)
        audio_path = os.path.join(self.test_dir, f"test_audio_{language}.wav")
        
        # Create a simple sine wave
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        
        # Save as WAV file
        import torchaudio
        torchaudio.save(audio_path, waveform, sample_rate)
        
        return audio_path
    
    @unittest.skip("Skipping integration test that requires model downloads")
    def test_process_voice_message(self):
        """Test the entire voice message processing pipeline"""
        # This test is skipped by default as it would require downloading models
        # and would be resource-intensive
        
        # Process the test audio
        result = process_voice_message(self.english_audio)
        
        # Check that we get a result dictionary
        self.assertIsInstance(result, dict)
        
        # Check that it contains the expected keys
        self.assertIn('detected_language', result)
        self.assertIn('transcription', result)
        self.assertIn('booking_details', result)
        self.assertIn('status', result)
        
    def test_module_initialization(self):
        """Test that all modules can be initialized"""
        try:
            # Try initializing all modules with minimal models
            lid = LanguageIdentifier(model_name="tiny")
            asr = SpeechRecognizer(model_size="tiny")
            intent = IntentDetector()
            
            # Check that initialization succeeded
            self.assertIsNotNone(lid)
            self.assertIsNotNone(asr)
            self.assertIsNotNone(intent)
        except Exception as e:
            self.fail(f"Module initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
