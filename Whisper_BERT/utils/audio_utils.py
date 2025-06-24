#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audio Utilities Module

This module provides helper functions for audio processing and handling.
"""

import os
import torch
import torchaudio
import numpy as np
from scipy import signal
import whisper
import subprocess
import shutil
import sys
import ssl
import platform
import tempfile

def find_ffmpeg():
    """Find the ffmpeg executable in the system."""
    try:
        if platform.system() == "Windows":
            # On Windows, check the PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
        else:
            # On Unix-like systems, use which command
            ffmpeg_path = subprocess.check_output(["which", "ffmpeg"]).decode().strip()
        return ffmpeg_path
    except (subprocess.SubprocessError, FileNotFoundError):
        # Try common locations
        common_paths = [
            "/usr/local/bin/ffmpeg",  # Homebrew on macOS
            "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
            "/usr/bin/ffmpeg",  # Most Linux distributions
            "/opt/local/bin/ffmpeg"   # MacPorts
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        print("Warning: ffmpeg not found. Audio processing may fail.")
        return None

def convert_audio_format(audio_path, target_format=".wav", target_sr=16000):
    """Convert audio to the target format."""
    output_path = os.path.splitext(audio_path)[0] + target_format
    ffmpeg_path = find_ffmpeg()
    
    if ffmpeg_path is None:
        print("Error: ffmpeg not found, cannot convert audio.")
        return audio_path
        
    try:
        subprocess.run([
            ffmpeg_path, "-y", "-i", audio_path, 
            "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1", 
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully converted {audio_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio: {e}")
        return audio_path

def preprocess_audio(audio_path, target_sr=16000):
    """
    Preprocess audio file for model input.
    
    Args:
        audio_path (str): Path to the audio file
        target_sr (int): Target sample rate
        
    Returns:
        str: Path to the preprocessed audio file
    """
    print(f"Preprocessing audio file: {audio_path}")
    
    # Fix SSL certificate issues
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get file extension
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    # Convert non-WAV formats to WAV for better compatibility
    if file_ext not in ['.wav']:
        audio_path = convert_audio_format(audio_path)
    
    try:
        # Use whisper's audio loading function for compatibility
        audio = whisper.load_audio(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error using whisper.load_audio: {e}")
        
        # Fallback to manual processing with torchaudio
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Create a temporary file for the preprocessed audio
            temp_path = audio_path.replace('.', '_processed.')
            torchaudio.save(temp_path, waveform, target_sr)
            return temp_path
            
        except Exception as e:
            print(f"Error preprocessing audio with torchaudio: {e}")
            return audio_path

def detect_silence(audio_path, threshold=0.01, min_silence_len=500):
    """
    Detect silent segments in audio.
    
    Args:
        audio_path (str): Path to the audio file
        threshold (float): Silence threshold
        min_silence_len (int): Minimum silence length in ms
        
    Returns:
        list: List of silent segments as (start, end) tuples in seconds
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to numpy
        audio_array = waveform.numpy().flatten()
        
        # Calculate amplitude
        amplitude = np.abs(audio_array)
        
        # Find silent regions
        is_silent = amplitude < threshold
        
        # Convert samples to ms
        samples_per_ms = sample_rate / 1000
        min_silence_samples = int(min_silence_len * samples_per_ms)
        
        # Find continuous silent regions
        silent_regions = []
        silent_start = None
        
        for i, silent in enumerate(is_silent):
            if silent and silent_start is None:
                silent_start = i
            elif not silent and silent_start is not None:
                if i - silent_start >= min_silence_samples:
                    # Convert to seconds
                    start_sec = silent_start / sample_rate
                    end_sec = i / sample_rate
                    silent_regions.append((start_sec, end_sec))
                silent_start = None
        
        # Check if the file ends with silence
        if silent_start is not None:
            if len(audio_array) - silent_start >= min_silence_samples:
                start_sec = silent_start / sample_rate
                end_sec = len(audio_array) / sample_rate
                silent_regions.append((start_sec, end_sec))
        
        return silent_regions
    
    except Exception as e:
        print(f"Error detecting silence: {e}")
        return []

def segment_audio(audio_path, output_dir, min_segment_len=1.0, max_segment_len=30.0):
    """
    Segment audio based on silence detection.
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save segmented audio files
        min_segment_len (float): Minimum segment length in seconds
        max_segment_len (float): Maximum segment length in seconds
        
    Returns:
        list: List of paths to segmented audio files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Detect silent regions
        silent_regions = detect_silence(audio_path)
        
        if not silent_regions:
            print("No silence detected for segmentation. Using the entire audio.")
            # Save the entire audio as a single segment
            segment_path = os.path.join(output_dir, "segment_0.wav")
            torchaudio.save(segment_path, waveform, sample_rate)
            return [segment_path]
        
        # Create segments based on silence
        segments = []
        audio_len = waveform.shape[1] / sample_rate
        
        segment_starts = [0]
        for start, end in silent_regions:
            # Use middle of silence as a segmentation point
            segment_point = (start + end) / 2
            segment_starts.append(segment_point)
        
        # Add end of audio
        segment_starts.append(audio_len)
        
        # Create segments
        segment_paths = []
        for i in range(len(segment_starts) - 1):
            start = segment_starts[i]
            end = segment_starts[i + 1]
            
            # Skip segments that are too short
            if end - start < min_segment_len:
                continue
            
            # Split segments that are too long
            if end - start > max_segment_len:
                num_splits = int(np.ceil((end - start) / max_segment_len))
                split_points = np.linspace(start, end, num_splits + 1)
                
                for j in range(len(split_points) - 1):
                    split_start = int(split_points[j] * sample_rate)
                    split_end = int(split_points[j + 1] * sample_rate)
                    
                    segment = waveform[:, split_start:split_end]
                    segment_path = os.path.join(output_dir, f"segment_{i}_{j}.wav")
                    torchaudio.save(segment_path, segment, sample_rate)
                    segment_paths.append(segment_path)
            else:
                # Create a single segment
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                
                segment = waveform[:, start_sample:end_sample]
                segment_path = os.path.join(output_dir, f"segment_{i}.wav")
                torchaudio.save(segment_path, segment, sample_rate)
                segment_paths.append(segment_path)
        
        return segment_paths
    
    except Exception as e:
        print(f"Error segmenting audio: {e}")
        return []