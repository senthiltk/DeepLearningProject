#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Module for Language Identification

This module contains test cases for the language identification module.
"""

import os
import sys
import unittest
import tempfile
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.language_identification import LanguageIdentifier

class TestLanguageIdentification(unittest.TestCase):
    """Test cases for Language Identification Module"""
    
    def setUp(self):
        """Set up test environment"""
        self.identifier = LanguageIdentifier(model_name="tiny")
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_initialize_model(self):
        """Test model initialization"""
        self.assertIsNotNone(self.identifier.model)
        self.assertIsInstance(self.identifier.model, object)
    
    def test_language_map(self):
        """Test language mapping"""
        langs = ['en', 'hi', 'ta', 'te', 'kn', 'ml']
        for lang in langs:
            self.assertIn(lang, self.identifier.language_map)
            name = self.identifier.get_language_name(lang)
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
    
    def test_unknown_language_handling(self):
        """Test handling of unknown language codes"""
        unknown_code = 'xyz'
        name = self.identifier.get_language_name(unknown_code)
        self.assertEqual(name, "Unknown")
    
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
