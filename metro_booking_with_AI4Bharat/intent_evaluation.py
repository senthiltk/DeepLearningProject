#!/usr/bin/env python3
"""
Intent Detection F1 Score Evaluation
Evaluates the performance of intent detection across all supported intent types
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language_processor import LanguageProcessor
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class IntentEvaluator:
    def __init__(self):
        self.processor = LanguageProcessor()
        
        # Test dataset for intent detection evaluation
        self.test_data = [
            # Book Ticket Intent
            ("Book a ticket from Majestic to MG Road", "book_ticket"),
            ("I want to book tickets from Indiranagar to Whitefield", "book_ticket"),
            ("Reserve a seat from Electronic City to Banashankari", "book_ticket"),
            ("Get me tickets for Jayanagar to Rajajinagar", "book_ticket"),
            ("Book 2 tickets from Airport to City Railway Station", "book_ticket"),
            ("मैजेस्टिक से एमजी रोड तक टिकट बुक करें", "book_ticket"),
            ("व्हाइटफील्ड जाना है", "book_ticket"),
            ("मुझे इंदिरानगर से बनशंकरी जाना है", "book_ticket"),
            ("ಮೆಜೆಸ್ಟಿಕ್ ನಿಂದ ಎಂ ಜಿ ರೋಡ್ ಗೆ ಟಿಕೆಟ್ ಬುಕ್ ಮಾಡಿ", "book_ticket"),
            ("ವೈಟ್‌ಫೀಲ್ಡ್ ಹೋಗಬೇಕು", "book_ticket"),
            ("மாஜஸ்டிக் இலிருந்து எம் ஜி ரோட் வரை டிக்கெட் புக் செய்யுங்கள்", "book_ticket"),
            ("వైట்‌ఫీల్డ్ వెళ್ళాలি", "book_ticket"),
            ("మెజెస్టిక్ నుండి ఎం జి రోడ్ వరకు టిక్కెట్ బుక్ చేయండి", "book_ticket"),
            ("मेजेस्टिक पासून एमजी रोड पर्यंत तिकीट बुक करा", "book_ticket"),
            ("व्हाईटफील्ड ला जाणे आहे", "book_ticket"),
            
            # Fare Inquiry Intent
            ("What is the fare from Majestic to MG Road?", "fare_inquiry"),
            ("How much does it cost from Indiranagar to Whitefield?", "fare_inquiry"),
            ("Tell me the price from Electronic City to Banashankari", "fare_inquiry"),
            ("What's the ticket price from Airport to City Railway?", "fare_inquiry"),
            ("How much money for Jayanagar to Rajajinagar?", "fare_inquiry"),
            ("मैजेस्टिक से एमजी रोड तक कितना पैसा लगेगा?", "fare_inquiry"),
            ("इंदिरानगर से बनशंकरी तक कितना खर्च होगा?", "fare_inquiry"),
            ("व्हाइटफील्ड तक का किराया क्या है?", "fare_inquiry"),
            ("ಮೆಜೆಸ್ಟಿಕ್ ನಿಂದ ಎಂ ಜಿ ರೋಡ್ ಎಷ್ಟು ಬೆಲೆ?", "fare_inquiry"),
            ("ಇಂದಿರಾನಗರ ನಿಂದ ಬನಶಂಕರಿ ಎಷ್ಟು ಹಣ?", "fare_inquiry"),
            ("மாஜஸ்டிக் இலிருந்து எம் ஜி ரோட் எவ்வளவு விலை?", "fare_inquiry"),
            ("இந்திரா நகர் இலிருந்து பனசங்கரி எவ்வளவு?", "fare_inquiry"),
            ("మెజెస్టిక్ నుండి ఎం జి రోడ్ ఎంత ధర?", "fare_inquiry"),
            ("ఇందిరానగర్ నుండి బనశంకరీ ఎంత?", "fare_inquiry"),
            ("मेजेस्टिक पासून एमजी रोड किती पैसे?", "fare_inquiry"),
            ("इंदिरानगर पासून बानशंकरी पर्यंत किती पैसे लागतील?", "fare_inquiry"),
            
            # Cancel Ticket Intent
            ("Cancel my ticket booking", "cancel_ticket"),
            ("I want to cancel my reservation", "cancel_ticket"),
            ("Please cancel the ticket I booked", "cancel_ticket"),
            ("Remove my booking from Majestic to MG Road", "cancel_ticket"),
            ("मेरा टिकट रद्द करें", "cancel_ticket"),
            ("मेरी बुकिंग कैंसल करना है", "cancel_ticket"),
            ("ನನ್ನ ಟಿಕೆಟ್ ರದ್ದು ಮಾಡಿ", "cancel_ticket"),
            ("என் டிக்கெட் ரத்து செய்யுங்கள்", "cancel_ticket"),
            ("నా టిక్కెట్ రద్దు చేయండి", "cancel_ticket"),
            ("माझे तिकीट रद्द करा", "cancel_ticket"),
            
            # Booking Status Intent
            ("What is my booking status?", "booking_status"),
            ("Check my ticket status", "booking_status"),
            ("Show my reservation details", "booking_status"),
            ("मेरी बुकिंग का स्टेटस क्या है?", "booking_status"),
            ("ನನ್ನ ಬುಕಿಂಗ್ ಸ್ಥಿತಿ ತೋರಿಸಿ", "booking_status"),
            ("என் பதிவு நிலை என்ன?", "booking_status"),
            ("నా బుకింగ్ స్థితి ఏమిటి?", "booking_status"),
            ("माझी बुकिंग स्थिती काय आहे?", "booking_status"),
            
            # Route Inquiry Intent
            ("How to reach Whitefield from Majestic?", "route_inquiry"),
            ("What is the route from Indiranagar to Banashankari?", "route_inquiry"),
            ("Show me the path to Electronic City", "route_inquiry"),
            ("Which line goes to MG Road?", "route_inquiry"),
            ("मैजेस्टिक से व्हाइटफील्ड कैसे जाएं?", "route_inquiry"),
            ("इंदिरानगर से बनशंकरी का रास्ता क्या है?", "route_inquiry"),
            ("ಮೆಜೆಸ್ಟಿಕ್ ನಿಂದ ವೈಟ್‌ಫೀಲ್ಡ್ ಹೇಗೆ ಹೋಗುವುದು?", "route_inquiry"),
            ("மாஜஸ்டிக் இலிருந்து வைட்ஃபீல்ட் எப்படி செல்வது?", "route_inquiry"),
            ("మెజెస్టిక్ నుండి వైట్‌ఫీల్డ్ ఎలా వెళ్ళాలి?", "route_inquiry"),
            ("मेजेस्टिक पासून व्हाईटफील्ड कसे जायचे?", "route_inquiry"),
            
            # General Inquiry Intent
            ("Help me with metro information", "general_inquiry"),
            ("I need help with the metro system", "general_inquiry"),
            ("What are the metro timings?", "general_inquiry"),
            ("Tell me about Bangalore Metro", "general_inquiry"),
            ("मेट्रो की जानकारी चाहिए", "general_inquiry"),
            ("मुझे मेट्रो के बारे में बताइए", "general_inquiry"),
            ("ಮೆಟ್ರೋ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು", "general_inquiry"),
            ("ಮೆಟ್ರೋ ಸಹಾಯ ಬೇಕು", "general_inquiry"),
            ("மெட்ரோ பற்றி தெரிந்து கொள்ள வேண்டும்", "general_inquiry"),
            ("மெட்ரோ தகவல் வேண்டும்", "general_inquiry"),
            ("మెట్రో గురించి తెలుసుకోవాలి", "general_inquiry"),
            ("మెట్రో సహాయం కావాలి", "general_inquiry"),
            ("मेट्रो बद्दल माहिती हवी", "general_inquiry"),
            ("मेट्रो मदत हवी", "general_inquiry"),
        ]
        
        self.intent_labels = [
            "book_ticket", "fare_inquiry", "cancel_ticket", 
            "booking_status", "route_inquiry", "general_inquiry"
        ]
    
    def evaluate_intent_detection(self):
        """Evaluate intent detection performance and calculate F1 scores"""
        print("🔍 Evaluating Intent Detection Performance...")
        print("=" * 60)
        
        y_true = []
        y_pred = []
        detailed_results = []
        
        for text, expected_intent in self.test_data:
            # Process the text
            result = self.processor.process_text(text)
            predicted_intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            y_true.append(expected_intent)
            y_pred.append(predicted_intent)
            
            detailed_results.append({
                'text': text,
                'expected': expected_intent,
                'predicted': predicted_intent,
                'confidence': confidence,
                'correct': expected_intent == predicted_intent
            })
            
            # Print individual results
            status = "✅" if expected_intent == predicted_intent else "❌"
            print(f"{status} Expected: {expected_intent}, Got: {predicted_intent} (conf: {confidence:.2f})")
            print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print()
        
        return y_true, y_pred, detailed_results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate detailed metrics including F1 scores"""
        print("\n📊 DETAILED PERFORMANCE METRICS")
        print("=" * 60)
        
        # Overall accuracy
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"🎯 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Classification report with F1 scores
        print("\n📈 CLASSIFICATION REPORT (F1 SCORES)")
        print("-" * 60)
        report = classification_report(y_true, y_pred, labels=self.intent_labels, 
                                     target_names=self.intent_labels, 
                                     zero_division=0, digits=3)
        print(report)
        
        # Individual F1 scores
        print("\n🎯 INDIVIDUAL F1 SCORES BY INTENT")
        print("-" * 40)
        for intent in self.intent_labels:
            f1 = f1_score(y_true, y_pred, labels=[intent], average='micro')
            print(f"{intent:20s}: {f1:.3f}")
        
        # Macro and Micro averages
        macro_f1 = f1_score(y_true, y_pred, labels=self.intent_labels, average='macro')
        micro_f1 = f1_score(y_true, y_pred, labels=self.intent_labels, average='micro')
        weighted_f1 = f1_score(y_true, y_pred, labels=self.intent_labels, average='weighted')
        
        print("\n📊 SUMMARY F1 SCORES")
        print("-" * 25)
        print(f"Macro F1:    {macro_f1:.3f}")
        print(f"Micro F1:    {micro_f1:.3f}")
        print(f"Weighted F1: {weighted_f1:.3f}")
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report
        }
    
    def generate_confusion_matrix(self, y_true, y_pred):
        """Generate and display confusion matrix"""
        print("\n🔍 CONFUSION MATRIX")
        print("-" * 30)
        
        cm = confusion_matrix(y_true, y_pred, labels=self.intent_labels)
        
        # Create DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=self.intent_labels, columns=self.intent_labels)
        
        print("Raw Confusion Matrix:")
        print(cm_df)
        
        print("\nNormalized Confusion Matrix (by true label):")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized_df = pd.DataFrame(cm_normalized, index=self.intent_labels, columns=self.intent_labels)
        print(cm_normalized_df.round(3))
        
        return cm, cm_normalized
    
    def analyze_by_language(self, detailed_results):
        """Analyze performance by language"""
        print("\n🌍 PERFORMANCE BY LANGUAGE")
        print("-" * 35)
        
        language_mapping = {
            'en': 'English',
            'hi': 'Hindi', 
            'kn': 'Kannada',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi'
        }
        
        language_stats = {}
        for result in detailed_results:
            # Detect language of the text
            lang_result = self.processor.detect_language(result['text'])
            # Handle both dict and string returns
            if isinstance(lang_result, dict):
                lang = lang_result.get('language', 'unknown')
            else:
                lang = lang_result
            
            if lang not in language_stats:
                language_stats[lang] = {'correct': 0, 'total': 0}
            
            language_stats[lang]['total'] += 1
            if result['correct']:
                language_stats[lang]['correct'] += 1
        
        for lang, stats in language_stats.items():
            lang_name = language_mapping.get(lang, lang)
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{lang_name:10s}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    def analyze_by_intent(self, detailed_results):
        """Analyze performance by intent type"""
        print("\n🎯 PERFORMANCE BY INTENT TYPE")
        print("-" * 35)
        
        intent_stats = {}
        for result in detailed_results:
            intent = result['expected']
            if intent not in intent_stats:
                intent_stats[intent] = {'correct': 0, 'total': 0}
            
            intent_stats[intent]['total'] += 1
            if result['correct']:
                intent_stats[intent]['correct'] += 1
        
        for intent in self.intent_labels:
            if intent in intent_stats:
                stats = intent_stats[intent]
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{intent:20s}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    def run_evaluation(self):
        """Run complete evaluation and generate report"""
        print("🧪 INTENT DETECTION F1 SCORE EVALUATION")
        print("=" * 60)
        print(f"📝 Test Cases: {len(self.test_data)}")
        print(f"🎯 Intent Types: {len(self.intent_labels)}")
        print("🌍 Languages: English, Hindi, Kannada, Tamil, Telugu, Marathi")
        print("=" * 60)
        
        # Run evaluation
        y_true, y_pred, detailed_results = self.evaluate_intent_detection()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Generate confusion matrix
        _, _ = self.generate_confusion_matrix(y_true, y_pred)
        
        # Language-wise analysis
        self.analyze_by_language(detailed_results)
        
        # Intent-wise analysis
        self.analyze_by_intent(detailed_results)
        
        # Summary
        print("\n🏆 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"📊 Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"🎯 Macro F1 Score:   {metrics['macro_f1']:.3f}")
        print(f"⚡ Micro F1 Score:   {metrics['micro_f1']:.3f}")
        print(f"⚖️  Weighted F1:      {metrics['weighted_f1']:.3f}")
        
        return metrics, detailed_results

if __name__ == "__main__":
    evaluator = IntentEvaluator()
    metrics, results = evaluator.run_evaluation()
