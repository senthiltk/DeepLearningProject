#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metro Booking Assistant Evaluation Module

This module evaluates the performance of the complete pipeline by processing
multiple audio files listed in a CSV along with their ground truth annotations.
"""

import os
import argparse
import torch
import json
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    import jiwer  # For Word Error Rate calculation
except ImportError:
    print("jiwer package not found. Installing...")
    import pip
    pip.main(['install', 'jiwer'])
    import jiwer

# Import modules from the project
from modules.language_identification import LanguageIdentifier
from modules.speech_recognition import SpeechRecognizer
from modules.intent_detection import IntentDetector
from utils.audio_utils import preprocess_audio

# Define supported languages
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te', 'kn', 'ml']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Metro Booking Assistant Pipeline')
    parser.add_argument('--csv', type=str, required=True, 
                      help='Path to the CSV file with audio paths and ground truth')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                      help='Path to save the evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run models on (cuda/cpu)')
    return parser.parse_args()

def process_audio_file(audio_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process a single audio file through the complete pipeline.
    
    Args:
        audio_path (str): Path to the audio file
        device (str): Device to run models on ('cuda' or 'cpu')
        
    Returns:
        dict: Extracted booking details
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file {audio_path} not found. Skipping.")
            return None
        
        # Step 1: Preprocess the audio
        preprocessed_audio = preprocess_audio(audio_path)
        
        # Step 2: Initialize language identifier and identify language
        language_identifier = LanguageIdentifier(device=device)
        detected_language = language_identifier.identify(preprocessed_audio)
        
        # Check if detected language is supported
        if detected_language not in SUPPORTED_LANGUAGES:
            detected_language = 'en'  # Default to English
        
        # Step 3: Transcribe speech using the detected language
        speech_recognizer = SpeechRecognizer(language=detected_language, device=device)
        transcription = speech_recognizer.transcribe(audio_path)
        
        # Step 4: Extract intent and booking details
        intent_detector = IntentDetector(language=detected_language, device=device)
        booking_details = intent_detector.extract_details(transcription)
        
        # Return results
        return {
            'detected_language': detected_language,
            'transcription': transcription,
            'source': booking_details.get('source', None),
            'destination': booking_details.get('destination', None),
            'tickets': booking_details.get('tickets', None),
        }
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def evaluate_pipeline(csv_path, device):
    """
    Evaluate the pipeline on multiple audio files with ground truth.
    
    Args:
        csv_path (str): Path to the CSV file with audio paths and ground truth
        device (str): Device to run models on ('cuda' or 'cpu')
        
    Returns:
        dict: Evaluation metrics and detailed results
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} entries")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    required_columns = ['audio_path', 'text', 'from_station', 'to_station', 'num_tickets']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            return None
    
    # Results storage
    results = []
    source_matches = []
    destination_matches = []
    tickets_matches = []
    all_predictions = []
    all_ground_truth = []
    
    # For Word Error Rate calculation
    all_reference_words = []
    all_hypothesis_words = []
    
    # For confusion matrices
    source_predictions = []
    source_truth = []
    destination_predictions = []
    destination_truth = []
    ticket_predictions = []
    ticket_truth = []
    
    # Process each audio file
    print(f"Processing {len(df)} audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row['audio_path']
        ground_truth = {
            'text': row['text'],
            'source': row['from_station'],
            'destination': row['to_station'],
            'tickets': row['num_tickets']
        }
        
        # Process the audio file
        prediction = process_audio_file(audio_path, device)
        
        if prediction is not None:
            # Convert ticket counts to integers for comparison if possible
            try:
                pred_tickets = int(prediction['tickets']) if prediction['tickets'] is not None else None
            except (ValueError, TypeError):
                pred_tickets = prediction['tickets']
                
            try:
                gt_tickets = int(ground_truth['tickets']) if ground_truth['tickets'] is not None else None
            except (ValueError, TypeError):
                gt_tickets = ground_truth['tickets']
            
            # Check if predictions match ground truth
            # Clean and compare source names, ignoring spaces and special tokens
            pred_source = prediction['source'].lower().replace(' ', '').replace('[sep]', '') if prediction['source'] else ''
            gt_source = ground_truth['source'].lower().replace(' ', '').replace('[sep]', '')
            source_match = (prediction['source'] and pred_source == gt_source)
            pred_destination = prediction['destination'].lower().replace(' ', '').replace('[sep]', '') if prediction['destination'] else ''
            gt_destination = ground_truth['destination'].lower().replace(' ', '').replace('[sep]', '')
            destination_match = (prediction['destination'] and pred_destination == gt_destination)
            tickets_match = (pred_tickets == gt_tickets)
            
            # Store match results
            source_matches.append(source_match)
            destination_matches.append(destination_match)
            tickets_matches.append(tickets_match)
            
            # Store combined entity match (all entities correct or not)
            all_predictions.append([source_match, destination_match, tickets_match])
            all_ground_truth.append([True, True, True])  # Ground truth is always true for all entities
            
            # Store data for confusion matrices
            source_predictions.append(prediction['source'] if prediction['source'] else "UNKNOWN")
            source_truth.append(ground_truth['source'])
            
            destination_predictions.append(prediction['destination'] if prediction['destination'] else "UNKNOWN")
            destination_truth.append(ground_truth['destination'])
            
            ticket_predictions.append(str(pred_tickets) if pred_tickets is not None else "UNKNOWN")
            ticket_truth.append(str(gt_tickets) if gt_tickets is not None else "UNKNOWN")
            
            # Store data for Word Error Rate (WER)
            if prediction['transcription'] and ground_truth['text']:
                ref_words = ground_truth['text'].lower().split()
                hyp_words = prediction['transcription'].lower().split()
                all_reference_words.append(ref_words)
                all_hypothesis_words.append(hyp_words)
            
            # Store detailed result
            result = {
                'audio_path': audio_path,
                'ground_truth': ground_truth,
                'prediction': {
                    'transcription': prediction['transcription'],
                    'source': prediction['source'],
                    'destination': prediction['destination'],
                    'tickets': prediction['tickets']
                },
                'matches': {
                    'source': source_match,
                    'destination': destination_match,
                    'tickets': tickets_match,
                    'all_correct': source_match and destination_match and tickets_match
                }
            }
            results.append(result)
        else:
            print(f"Warning: Failed to process {audio_path}. Skipping in evaluation.")
    
    # Calculate basic evaluation metrics
    metrics = {
        'source_accuracy': np.mean(source_matches) if source_matches else 0,
        'destination_accuracy': np.mean(destination_matches) if destination_matches else 0,
        'tickets_accuracy': np.mean(tickets_matches) if tickets_matches else 0,
        'overall_accuracy': np.mean([all(match) for match in zip(source_matches, destination_matches, tickets_matches)]) if source_matches else 0,
        'total_samples': len(df),
        'processed_samples': len(results)
    }
    
    # Calculate Word Error Rate (WER)
    if all_reference_words and all_hypothesis_words:
        total_wer = calculate_wer(all_reference_words, all_hypothesis_words)
        metrics['word_error_rate'] = total_wer
    
    # Generate confusion matrices
    if source_truth and source_predictions:
        source_confusion = generate_confusion_matrix(source_truth, source_predictions)
        metrics['source_confusion_matrix'] = source_confusion
        
    if destination_truth and destination_predictions:
        destination_confusion = generate_confusion_matrix(destination_truth, destination_predictions)
        metrics['destination_confusion_matrix'] = destination_confusion
        
    if ticket_truth and ticket_predictions:
        tickets_confusion = generate_confusion_matrix(ticket_truth, ticket_predictions)
        metrics['tickets_confusion_matrix'] = tickets_confusion
    
    # Create detailed evaluation report
    evaluation = {
        'metrics': metrics,
        'detailed_results': results
    }
    
    return evaluation

def calculate_wer(references, hypotheses):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts.
    
    Args:
        references (list): List of reference word lists
        hypotheses (list): List of hypothesis word lists
        
    Returns:
        float: Word Error Rate
    """
    
    # Calculate WER for each pair
    wer_scores = []
    for ref, hyp in zip(references, hypotheses):
        # Convert lists to strings if needed
        if isinstance(ref, list):
            ref = ' '.join(ref)
        if isinstance(hyp, list):
            hyp = ' '.join(hyp)
            
        # Calculate WER for this pair
        try:
            score = jiwer.wer(ref, hyp)
            wer_scores.append(score)
        except Exception as e:
            print(f"Error calculating WER: {e}")
            # Skip this pair
            continue
    
    # Return average WER
    return np.mean(wer_scores) if wer_scores else 1.0

def generate_confusion_matrix(y_true, y_pred, max_classes=10):
    """
    Generate a confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        max_classes (int): Maximum number of classes to include in matrix
        
    Returns:
        dict: Confusion matrix data
    """
    
    # Get unique classes from both true and predicted labels
    classes = sorted(list(set(y_true + y_pred)))
    
    # Limit the number of classes if there are too many
    if len(classes) > max_classes:
        # Count occurrences of each class
        counter = {}
        for c in y_true:
            counter[c] = counter.get(c, 0) + 1
        
        # Select the most frequent classes
        top_classes = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)[:max_classes-1]
        if "UNKNOWN" not in top_classes:
            top_classes.append("UNKNOWN")
        
        # Map other classes to "Other"
        y_true_filtered = ["Other" if y not in top_classes else y for y in y_true]
        y_pred_filtered = ["Other" if y not in top_classes and y != "UNKNOWN" else y for y in y_pred]
        
        classes = sorted(list(set(y_true_filtered + y_pred_filtered)))
        
        # Calculate confusion matrix with filtered classes
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=classes)
    else:
        # Calculate confusion matrix with all classes
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Convert to list for JSON serialization
    cm_list = cm.tolist()
    
    return {
        'matrix': cm_list,
        'classes': classes
    }
def print_metrics_summary(metrics):
    """Print a summary of the evaluation metrics."""
    print("\n===== Evaluation Results =====")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Successfully processed: {metrics['processed_samples']}")
    
    print("\nEntity Recognition Accuracy:")
    print(f"  Source Station: {metrics['source_accuracy']:.4f}")
    print(f"  Destination Station: {metrics['destination_accuracy']:.4f}")
    print(f"  Number of Tickets: {metrics['tickets_accuracy']:.4f}")
    
    print(f"\nOverall Accuracy (all entities correct): {metrics['overall_accuracy']:.4f}")
    
    # Print Word Error Rate if available
    if 'word_error_rate' in metrics:
        print(f"\nTranscription Word Error Rate: {metrics['word_error_rate']:.4f}")
        print(f"(Lower WER is better: 0 = perfect, 1 = completely wrong)")
    
    # Print confusion matrix summaries if available
    if 'source_confusion_matrix' in metrics:
        print("\nSource Station Confusion (Top 5):")
        print_confusion_summary(metrics['source_confusion_matrix'], top_n=5)
        
    if 'destination_confusion_matrix' in metrics:
        print("\nDestination Station Confusion (Top 5):")
        print_confusion_summary(metrics['destination_confusion_matrix'], top_n=5)
        
    if 'tickets_confusion_matrix' in metrics:
        print("\nTicket Count Confusion (Top 5):")
        print_confusion_summary(metrics['tickets_confusion_matrix'], top_n=5)
    
    print("=============================")

def print_confusion_summary(confusion_data, top_n=5):
    """Print a summary of the confusion matrix showing top confused pairs."""
    matrix = np.array(confusion_data['matrix'])
    classes = confusion_data['classes']
    
    # Find pairs with highest confusion (excluding correct predictions)
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and matrix[i, j] > 0:
                confused_pairs.append((classes[i], classes[j], matrix[i, j]))
    
    # Sort pairs by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Print top N confused pairs
    for true_label, pred_label, count in confused_pairs[:top_n]:
        print(f"  True: '{true_label}' → Predicted: '{pred_label}' ({count} times)")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found.")
        return
    
    # Evaluate the pipeline
    evaluation = evaluate_pipeline(args.csv, args.device)
    
    if evaluation:
        # Print metrics summary
        print_metrics_summary(evaluation['metrics'])
        
        # Save detailed evaluation results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed results saved to {args.output}")
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()