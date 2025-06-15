#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intent Detection Module

This module is responsible for identifying the booking intent and extracting relevant
details from the transcribed text, such as source station, destination station,
number of tickets, and travel time.
"""

import os
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForTokenClassification

class IntentDetector:
    """
    Class to detect booking intent and extract relevant entities from transcribed text.
    """
    
    def __init__(self, language='en', model_type='bert', device=None):
        """
        Initialize the intent detection model.
        
        Args:
            language (str): Language code for intent detection
            model_type (str): Type of model to use ('bert', 'roberta', 'xlm-roberta')
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
            'bert': {
                'en': "distilbert-base-uncased",
                'hi': "ai4bharat/indic-bert",
                'multilingual': "bert-base-multilingual-uncased"
            },
            'roberta': {
                'en': "roberta-base",
                'multilingual': "xlm-roberta-base"
            }
        }
        
        # Initialize intent classification model
        self._initialize_intent_model()
        
        # Initialize named entity recognition (NER) model for entity extraction
        self._initialize_ner_model()
        
        # Load predefined station names (simulated)
        self.stations = self._load_stations()
    
    def _initialize_intent_model(self):
        """Initialize intent classification model."""
        try:
            # For a real implementation, you would fine-tune these models on metro booking data
            # Here we're using a pretrained model as a placeholder
            
            if self.language == 'en':
                model_id = self.model_map[self.model_type]['en']
            elif self.language in ['hi', 'ta', 'te', 'kn', 'ml']:
                # Use multilingual model for Indian languages
                model_id = self.model_map[self.model_type].get('multilingual', 
                                                             self.model_map['bert']['multilingual'])
            else:
                model_id = self.model_map['bert']['multilingual']
            
            print(f"Loading intent detection model: {model_id}")
            
            # In a real implementation, you would load a fine-tuned model like:
            # self.tokenizer = AutoTokenizer.from_pretrained(f"./models/intent_{self.language}")
            # self.model = AutoModelForSequenceClassification.from_pretrained(f"./models/intent_{self.language}")
            
            # For our demonstration, we'll use a text classification pipeline with a base model
            # that would typically be fine-tuned for this task
            self.intent_pipeline = pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            print(f"Error loading intent model: {e}")
            print("Using rule-based fallback for intent detection")
            self.intent_pipeline = None
    
    def _initialize_ner_model(self):
        """Initialize named entity recognition model for entity extraction."""
        try:
            # In a real implementation, you would load a fine-tuned NER model
            # For now, we'll use a pre-trained model and supplement with rule-based extraction
            
            if self.language == 'en':
                model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
            else:
                # Use multilingual model for other languages
                model_id = "xlm-roberta-large-finetuned-conll03-english"
            
            print(f"Loading NER model: {model_id}")
            
            self.ner_pipeline = pipeline(
                "token-classification",
                model=model_id,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            print(f"Error loading NER model: {e}")
            print("Using rule-based fallback for entity extraction")
            self.ner_pipeline = None
    
    def _load_stations(self):
        """
        Load predefined station names.
        
        In a real implementation, this would load from a database or a file.
        For demonstration purposes, we're using a hardcoded dictionary.
        """
        # Simulated metro station data (for demonstration purposes)
        stations = {
            'en': [
                "central", "mg road", "indiranagar", "majestic", "vidhana soudha",
                "cubbon park", "lal bagh", "jayanagar", "banashankari", "jp nagar",
                "electronic city", "whitefield", "airport", "bus station", "railway station"
            ],
            'hi': [
                "सेंट्रल", "एमजी रोड", "इंदिरानगर", "मजेस्टिक", "विधान सौधा",
                "कब्बन पार्क", "लाल बाग", "जयनगर", "बनशंकरी", "जेपी नगर",
                "इलेक्ट्रॉनिक सिटी", "व्हाइटफील्ड", "हवाई अड्डा", "बस स्टेशन", "रेलवे स्टेशन"
            ],
            # Add entries for other languages as needed
        }
        
        # If language not explicitly supported, fall back to English
        if self.language not in stations:
            return stations['en']
        
        return stations[self.language]
    
    def detect_intent(self, text):
        """
        Detect if the text contains a metro booking intent.
        
        Args:
            text (str): The transcribed text
        
        Returns:
            bool: True if booking intent is detected, False otherwise
        """
        # In a real implementation, this would use the intent classification model
        # For demonstration, we'll use a simple keyword-based approach
        
        if self.intent_pipeline is not None:
            # This is a placeholder. In a real implementation, you would:
            # 1. Fine-tune the model on labeled data with intents like "book_metro", "check_schedule", etc.
            # 2. Use the model's prediction to determine the intent
            
            # Since we haven't fine-tuned the model, we'll combine with a rule-based approach
            result = self.intent_pipeline(text)
            print(f"Intent detection result (placeholder): {result}")
        
        # Rules-based intent detection as fallback or supplement
        booking_keywords = {
            'en': ['book', 'ticket', 'metro', 'train', 'from', 'to', 'station'],
            'hi': ['बुक', 'टिकट', 'मेट्रो', 'ट्रेन', 'से', 'तक', 'स्टेशन'],
            'ta': ['புக்', 'டிக்கெட்', 'மெட்ரோ', 'ரயில்', 'இருந்து', 'வரை', 'நிலையம்'],
            # Add keywords for other languages as needed
        }
        
        # Get keywords for the current language, fallback to English if not available
        keywords = booking_keywords.get(self.language, booking_keywords['en'])
        
        # Check if any keyword is in the text
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return True
        
        # If no booking keywords were found, it's likely not a booking intent
        return False
    
    def extract_entities(self, text):
        """
        Extract named entities from text.
        
        Args:
            text (str): The transcribed text
        
        Returns:
            dict: Extracted entities (source, destination, tickets, time)
        """
        entities = {
            'source': None,
            'destination': None, 
            'tickets': 1,  # Default value
            'time': None
        }
        
        # Try model-based NER first if available
        if self.ner_pipeline is not None:
            ner_results = self.ner_pipeline(text)
            print(f"NER results: {ner_results}")
            
            # In a fine-tuned model, we would expect custom entity labels like SOURCE, DESTINATION
            # Since we're using a generic model, we'll supplement with rule-based extraction
        
        # Rule-based entity extraction
        self._extract_stations(text, entities)
        self._extract_tickets(text, entities)
        self._extract_time(text, entities)
        
        return entities
    
    def _extract_stations(self, text, entities):
        """Extract source and destination stations using rule-based methods."""
        text_lower = text.lower()
        
        # Define patterns based on language
        if self.language == 'en':
            from_patterns = [r'from\s+(\w+(?:\s+\w+)*)', r'at\s+(\w+(?:\s+\w+)*)', r'starting\s+from\s+(\w+(?:\s+\w+)*)']
            to_patterns = [r'to\s+(\w+(?:\s+\w+)*)', r'towards\s+(\w+(?:\s+\w+)*)', r'arriving\s+at\s+(\w+(?:\s+\w+)*)']
        elif self.language == 'hi':
            from_patterns = [r'से\s+(\w+(?:\s+\w+)*)', r'फ्रॉम\s+(\w+(?:\s+\w+)*)']
            to_patterns = [r'को\s+(\w+(?:\s+\w+)*)', r'तक\s+(\w+(?:\s+\w+)*)', r'टू\s+(\w+(?:\s+\w+)*)']
        else:
            # Generic patterns - less accurate but better than nothing
            from_patterns = [r'from\s+(\w+(?:\s+\w+)*)', r'at\s+(\w+(?:\s+\w+)*)']
            to_patterns = [r'to\s+(\w+(?:\s+\w+)*)', r'towards\s+(\w+(?:\s+\w+)*)']
        
        # Try to extract source station
        for pattern in from_patterns:
            matches = re.search(pattern, text_lower)
            if matches and matches.group(1):
                entities['source'] = matches.group(1)
                break
        
        # Try to extract destination station
        for pattern in to_patterns:
            matches = re.search(pattern, text_lower)
            if matches and matches.group(1):
                entities['destination'] = matches.group(1)
                break
        
        # Match extracted stations with known station names
        if entities['source']:
            entities['source'] = self._match_station(entities['source'])
        
        if entities['destination']:
            entities['destination'] = self._match_station(entities['destination'])
    
    def _extract_tickets(self, text, entities):
        """Extract number of tickets using rule-based methods."""
        text_lower = text.lower()
        
        # Define patterns based on language
        if self.language == 'en':
            patterns = [
                r'(\d+)\s+tickets?', r'tickets?\s+(\d+)',
                r'book\s+(\d+)', r'(\d+)\s+persons?'
            ]
        elif self.language == 'hi':
            patterns = [
                r'(\d+)\s+टिकट', r'टिकट\s+(\d+)',
                r'बुक\s+(\d+)', r'(\d+)\s+लोगों'
            ]
        else:
            # Generic patterns
            patterns = [r'(\d+)\s+tickets?', r'tickets?\s+(\d+)']
        
        # Try to extract number of tickets
        for pattern in patterns:
            matches = re.search(pattern, text_lower)
            if matches and matches.group(1):
                try:
                    entities['tickets'] = int(matches.group(1))
                    break
                except ValueError:
                    continue
    
    def _extract_time(self, text, entities):
        """Extract time using rule-based methods."""
        text_lower = text.lower()
        
        # Define patterns based on language
        if self.language == 'en':
            patterns = [
                r'at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)',
                r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))',
                r'(\d{1,2})\s+o\'?clock'
            ]
        elif self.language == 'hi':
            patterns = [
                r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+बजे',
                r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+समय'
            ]
        else:
            # Generic patterns
            patterns = [r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)']
        
        # Try to extract time
        for pattern in patterns:
            matches = re.search(pattern, text_lower)
            if matches and matches.group(1):
                entities['time'] = matches.group(1)
                break
    
    def _match_station(self, extracted_name):
        """
        Match extracted station name with known station names.
        
        Args:
            extracted_name (str): The extracted station name
        
        Returns:
            str: The matched station name or the original name if no match
        """
        extracted_name = extracted_name.lower()
        
        # Simple fuzzy matching - in a real implementation, use a proper fuzzy match algorithm
        best_match = None
        best_score = 0
        
        for station in self.stations:
            station_lower = station.lower()
            
            # Check if the station name contains the extracted name or vice versa
            if extracted_name in station_lower or station_lower in extracted_name:
                score = len(set(extracted_name) & set(station_lower)) / max(len(extracted_name), len(station_lower))
                
                if score > best_score:
                    best_score = score
                    best_match = station
        
        # If we found a reasonable match, use it
        if best_match and best_score > 0.5:
            return best_match
        
        # Otherwise, return the original extracted name
        return extracted_name
    
    def extract_details(self, text):
        """
        Extract booking details from transcribed text.
        
        Args:
            text (str): The transcribed text
        
        Returns:
            dict: Extracted booking details (source, destination, tickets, time)
        """
        # Check if the text contains a booking intent
        has_booking_intent = self.detect_intent(text)
        
        if not has_booking_intent:
            print("No booking intent detected in the text.")
            return {}
        
        # Extract entities for booking
        entities = self.extract_entities(text)
        
        # Post-process and validate the extracted entities
        # (In a real implementation, this would involve additional validation and error handling)
        
        return entities

# Testing function
def test_intent_detection(text, language='en'):
    """Test the intent detection module with a sample text."""
    detector = IntentDetector(language=language)
    booking_details = detector.extract_details(text)
    
    print(f"Input text: {text}")
    print(f"Extracted booking details: {booking_details}")
    
    return booking_details

if __name__ == "__main__":
    # Test with a sample text
    import sys
    
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        language = sys.argv[2] if len(sys.argv) > 2 else 'en'
        
        test_intent_detection(input_text, language)
    else:
        # Example text for testing
        example_text = "Book one ticket from Majestic to MG Road at 5:30pm"
        test_intent_detection(example_text)
