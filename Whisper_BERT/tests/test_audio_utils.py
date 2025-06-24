#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Module for Audio Utilities

This module contains test cases for the audio utility functions.
"""

import os
import sys
import unittest
import tempfile
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import preprocess_audio, detect_silence, segment_audio

class TestAudioUtils(unittest.TestCase):
    """Test cases for Audio Utility Module"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test audio file
        self.test_audio = self.create_test_audio()
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.test_dir)
    
    def create_test_audio(self):
        """Create a test audio file with silence and non-silence sections"""
        audio_path = os.path.join(self.test_dir, "test_audio.wav")
        
        # Create a 5-second audio with 1 second of silence in the middle
        sample_rate = 16000
        duration = 5
        
        # Generate time points
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create sine wave for non-silent parts
        waveform = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave
        
        # Insert silence in the middle (between 2s and 3s)
        silence_start = int(2 * sample_rate)
        silence_end = int(3 * sample_rate)
        waveform[silence_start:silence_end] = 0.0
        
        # Reshape to match torchaudio's expected format [channels, samples]
        waveform = waveform.unsqueeze(0)
        
        # Save as WAV file
        import torchaudio
        torchaudio.save(audio_path, waveform, sample_rate)
        
        return audio_path
    
    def test_preprocess_audio(self):
        """Test audio preprocessing"""
        # Process the audio
        processed_path = preprocess_audio(self.test_audio)
        
        # Check that the processed path exists
        self.assertTrue(os.path.exists(processed_path))
        
        # Check that processing doesn't fail with non-existent file
        with self.assertRaises(FileNotFoundError):
            preprocess_audio("non_existent_file.wav")
    
    def test_detect_silence(self):
        """Test silence detection"""
        # Detect silence in the test audio
        silence_regions = detect_silence(self.test_audio)
        
        # We should have at least one silent region
        self.assertGreater(len(silence_regions), 0)
        
        # Check that each region is a tuple of (start, end)
        for region in silence_regions:
            self.assertIsInstance(region, tuple)
            self.assertEqual(len(region), 2)
            start, end = region
            self.assertLess(start, end)
    
    def test_segment_audio(self):
        """Test audio segmentation"""
        # Create output directory for segments
        segment_dir = os.path.join(self.test_dir, "segments")
        
        # Segment the audio
        segment_paths = segment_audio(self.test_audio, segment_dir)
        
        # We should have at least one segment
        self.assertGreater(len(segment_paths), 0)
        
        # Check that each segment file exists
        for path in segment_paths:
            self.assertTrue(os.path.exists(path))
            
            # Check that each segment is a valid audio file
            import torchaudio
            try:
                waveform, sample_rate = torchaudio.load(path)
                self.assertGreater(waveform.shape[1], 0)  # Should have samples
                self.assertGreater(sample_rate, 0)  # Should have a valid sample rate
            except Exception as e:
                self.fail(f"Failed to load segment {path}: {e}")

if __name__ == '__main__':
    unittest.main()
