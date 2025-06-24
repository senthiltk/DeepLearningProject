#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metro Booking Voice Assistant - Main Module

This is the main entry point for the Metro Booking Voice Assistant application.
It coordinates the language identification, speech recognition, 
and intent detection modules to process voice messages.
"""

import os
import argparse
import torch
import json
import ssl
import sys
import platform
import subprocess

# Fix SSL certificate verification issue (macOS specific)
if platform.system() == 'Darwin':
    # Disable SSL certificate verification for macOS
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Add path to ffmpeg if it's not found
    try:
        ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
        print(f"Found ffmpeg at: {ffmpeg_path}")
    except subprocess.CalledProcessError:
        # Try common Homebrew paths
        homebrew_ffmpeg_paths = [
            '/usr/local/bin',
            '/opt/homebrew/bin',
            '/opt/local/bin'
        ]
        for path in homebrew_ffmpeg_paths:
            if os.path.exists(os.path.join(path, 'ffmpeg')):
                os.environ["PATH"] += os.pathsep + path
                print(f"Added ffmpeg path: {path}")
                break

from modules.language_identification import LanguageIdentifier
from modules.speech_recognition import SpeechRecognizer
from modules.intent_detection import IntentDetector
from utils.audio_utils import preprocess_audio

# Define supported languages
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te', 'kn', 'ml']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metro Booking Voice Assistant')
    parser.add_argument('--audio', type=str, required=True, 
                      help='Path to the audio file')
    parser.add_argument('--output', type=str, default='booking_result.json',
                      help='Path to save the booking details')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run models on (cuda/cpu)')
    return parser.parse_args()

def process_voice_message(audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process a voice message to extract booking intent and details.
    
    Args:
        audio_path (str): Path to the audio file
        device (str): Device to run models on ('cuda' or 'cpu')
        
    Returns:
        dict: Extracted booking details
    """
    print(f"Processing audio file: {audio_path} on {device}")
    
    # Step 1: Preprocess the audio
    preprocessed_audio = preprocess_audio(audio_path)
    
    # Step 2: Initialize language identifier and identify language
    language_identifier = LanguageIdentifier(device=device)
    detected_language = language_identifier.identify(preprocessed_audio)
    print(f"Detected language: {detected_language}")
    
    # Check if detected language is supported
    if detected_language not in SUPPORTED_LANGUAGES:
        print(f"Warning: Detected language {detected_language} is not in supported languages list.")
        print(f"Defaulting to English for further processing.")
        detected_language = 'en'
    
    # Step 3: Transcribe speech using the detected language
    speech_recognizer = SpeechRecognizer(language=detected_language, device=device)
    transcription = speech_recognizer.transcribe(audio_path)
    print(f"Transcription: {transcription}")
    
    # Step 4: Extract intent and booking details
    intent_detector = IntentDetector(language=detected_language, device=device)
    booking_details = intent_detector.extract_details(transcription)
    
    # Step 5: Add metadata to the results
    result = {
        'detected_language': detected_language,
        'transcription': transcription,
        'booking_details': booking_details,
        'status': 'success' if all(k in booking_details for k in ['source', 'destination']) else 'incomplete'
    }
    
    return result

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} not found.")
        return
        
    # Check file extension and convert if needed
    file_ext = os.path.splitext(args.audio)[1].lower()
    temp_audio_path = args.audio
    
    if file_ext in ['.m4a', '.mp4', '.aac', '.mp3']:
        print(f"Converting {file_ext} file to WAV format for better compatibility...")
        try:
            temp_audio_path = os.path.splitext(args.audio)[0] + '.wav'
            subprocess.run(['ffmpeg', '-y', '-i', args.audio, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path], 
                         check=True, 
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
            print(f"Converted to: {temp_audio_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not convert audio file: {e}")
            # Continue with original file
    
    # Process the voice message
    result = process_voice_message(temp_audio_path, args.device)
    
    # Save the results to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print booking summary
    print("\n===== Booking Summary =====")
    print(f"Language: {result['detected_language']}")
    print(f"Source: {result['booking_details'].get('source', 'Not detected')}")
    print(f"Destination: {result['booking_details'].get('destination', 'Not detected')}")
    print(f"Number of tickets: {result['booking_details'].get('tickets', 1)}")
    print(f"Time: {result['booking_details'].get('time', 'Not specified')}")
    print("==========================\n")
    
    if result['status'] == 'incomplete':
        print("Warning: Some required booking details are missing. Please provide complete information.")
    
if __name__ == "__main__":
    main()
