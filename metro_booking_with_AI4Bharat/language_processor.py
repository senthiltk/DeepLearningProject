"""
Language Processing Module with Enhanced Indian Language Support
Intent detection and information extraction for metro booking
Supports: English, Hindi, Kannada, Tamil, Telugu, Marathi
Features: Automatic language detection, romanized text transliteration
"""

import re
import json
from typing import Dict, List, Any, Optional
from difflib import get_close_matches

# Import transliteration libraries
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    print("Warning: indic-transliteration not available. Native script display will be limited.")

class LanguageProcessor:
    """Process text for intent detection and entity extraction with Indian language support"""
    
    def __init__(self):
        # Language detection patterns
        self.script_patterns = {
            'kn': r'[\u0C80-\u0CFF]',     # Kannada script
            'ta': r'[\u0B80-\u0BFF]',     # Tamil script
            'te': r'[\u0C00-\u0C7F]',     # Telugu script
            'devanagari': r'[\u0900-\u097F]',  # Devanagari (Hindi/Marathi)
        }
        
        # Key vocabulary to distinguish Hindi vs Marathi
        self.hindi_keywords = ['से', 'तक', 'टिकट', 'कितना', 'मदद', 'जाना', 'के लिए', 'पैसा', 'रुपया']
        self.marathi_keywords = ['पासून', 'पर्यंत', 'तिकीट', 'किती', 'मदत', 'जाणे', 'साठी', 'पैसे', 'रुपये']
        
        # Metro stations (comprehensive list)
        self.metro_stations = [
            "Majestic", "City Railway Station", "MG Road", "Cubbon Park",
            "Vidhana Soudha", "Indiranagar", "Banashankari", "Jayanagar",
            "Whitefield", "Electronic City", "Silk Board", "BTM Layout",
            "JP Nagar", "Bannerghatta Road", "Hosur Road", "Koramangala",
            "HSR Layout", "Rajajinagar", "Malleshwaram", "Yeshwantpur",
            "Hebbal", "Airport", "Yelahanka", "Nagasandra", "Peenya",
            "Vijayanagar", "Attiguppe", "Mysore Road", "Kengeri",
            "Rajarajeshwari Nagar", "Jnanabharathi", "Magadi Road",
            "Sandal Soap Factory", "Mahalakshmi", "Sampige Road",
            "Nadaprabhu Kempegowda Station", "Chickpet", "KR Market",
            "National College", "Lalbagh", "South End Circle",
            "Yelachenahalli", "Konanakunte Cross", "Doddakallasandra",
            "Gottigere", "Thalghattapura", "Vajarahalli", "Silk Institute",
            "Marathalli", "Kadugodi", "Channasandra", "Hoodi", "Garudacharpalya",
            "Domlur", "Swami Vivekananda Road", "Kalyan Nagar",
            "Nagawara", "Thanisandra", "Kempegowda International Airport"
        ]
        
        # Enhanced station aliases with transliterations
        self.station_aliases = {
            # English aliases
            "majestic": "Majestic",
            "kempegowda": "Majestic", 
            "railway station": "City Railway Station",
            "city station": "City Railway Station",
            "mg road": "MG Road",
            "brigade road": "MG Road",
            "cubbon park": "Cubbon Park",
            "vidhana soudha": "Vidhana Soudha",
            "assembly": "Vidhana Soudha",
            "indiranagar": "Indiranagar",
            "indira nagar": "Indiranagar",
            "banashankari": "Banashankari",
            "bsk": "Banashankari",
            "jayanagar": "Jayanagar",
            "jn": "Jayanagar",
            "whitefield": "Whitefield",
            "white field": "Whitefield",
            "electronic city": "Electronic City",
            "e city": "Electronic City",
            "silk board": "Silk Board",
            "btm": "BTM Layout",
            "btm layout": "BTM Layout",
            "jp nagar": "JP Nagar",
            "jaya prakash nagar": "JP Nagar",
            "bannerghatta": "Bannerghatta Road",
            "hosur road": "Hosur Road",
            "koramangala": "Koramangala",
            "hsr": "HSR Layout",
            "hsr layout": "HSR Layout",
            "rajajinagar": "Rajajinagar",
            "rr nagar": "Rajarajeshwari Nagar",
            "airport": "Kempegowda International Airport",
            "marathalli": "Marathalli",
            
            # Hindi transliterations
            "मैजेस्टिक": "Majestic",
            "मजेस्टिक": "Majestic",
            "केम्पेगौड़ा": "Majestic",
            "एम जी रोड": "MG Road",
            "एमजी रोड": "MG Road",
            "कब्बन पार्क": "Cubbon Park",
            "विधान सौध": "Vidhana Soudha",
            "इंदिरानगर": "Indiranagar",
            "इन्दिरानगर": "Indiranagar",
            "बनशंकरी": "Banashankari",
            "जयनगर": "Jayanagar",
            "व्हाइटफील्ड": "Whitefield",
            "व्हाइटफील्ड": "Whitefield",
            "इलेक्ट्रॉनिक सिटी": "Electronic City",
            "सिल्क बोर्ड": "Silk Board",
            "बीटीएम": "BTM Layout",
            "जेपी नगर": "JP Nagar",
            "बैनरघट्टा": "Bannerghatta Road",
            "होसूर रोड": "Hosur Road",
            "कोरमंगला": "Koramangala",
            "एचएसआर": "HSR Layout",
            "राजाजी नगर": "Rajajinagar",
            "एयरपोर्ट": "Kempegowda International Airport",
            "मारथली": "Marathalli",
            "मारथहल्ली": "Marathalli",
            
            # Marathi transliterations (enhanced)
            "मेजेस्टिक": "Majestic",
            "केम्पेगौडा": "Majestic",
            "एमजी रोड": "MG Road",
            "एम जी रोड": "MG Road",
            "कबन पार्क": "Cubbon Park",
            "कब्बन पार्क": "Cubbon Park",
            "विधान सौधा": "Vidhana Soudha",
            "इंदिरानगर": "Indiranagar",
            "इन्दिरानगर": "Indiranagar",
            "बानशंकरी": "Banashankari",
            "बनशंकरी": "Banashankari",
            "जयनगर": "Jayanagar",
            "व्हाईटफील्ड": "Whitefield",
            "व्हाइटफील्ड": "Whitefield",
            "इलेक्ट्रॉनिक सिटी": "Electronic City",
            "इलेक्ट्रानिक शहर": "Electronic City",
            "सिल्क बोर्ड": "Silk Board",
            "बीटीएम": "BTM Layout",
            "बीटीएम लेआउट": "BTM Layout",
            "जेपी नगर": "JP Nagar",
            "जयप्रकाश नगर": "JP Nagar",
            "बॅनरघट्टा": "Bannerghatta Road",
            "होसूर रोड": "Hosur Road",
            "कोरमंगला": "Koramangala",
            "एचएसआर": "HSR Layout",
            "एचएसआर लेआउट": "HSR Layout",
            "राजाजीनगर": "Rajajinagar",
            "राजाजी नगर": "Rajajinagar",
            "मराठहल्ली": "Marathalli",
            "मराठाहल्ली": "Marathalli",
            "एअरपोर्ट": "Kempegowda International Airport",
            "विमानतळ": "Kempegowda International Airport",
            
            # Kannada transliterations
            "ಮೆಜೆಸ್ಟಿಕ್": "Majestic",
            "ಎಂ ಜಿ ರೋಡ್": "MG Road",
            "ಕಬ್ಬನ್ ಪಾರ್ಕ್": "Cubbon Park",
            "ವಿಧಾನ ಸೌಧ": "Vidhana Soudha",
            "ಇಂದಿರಾನಗರ": "Indiranagar",
            "ಬನಶಂಕರಿ": "Banashankari",
            "ಜಯನಗರ": "Jayanagar",
            "ವೈಟ್‌ಫೀಲ್ಡ್": "Whitefield",
            "ಮರಾಠಹಳ್ಳಿ": "Marathalli",
            
            # Tamil transliterations
            "மாஜஸ்டிக்": "Majestic",
            "எம் ஜி ரோட்": "MG Road",
            "கப்பன் பார்க்": "Cubbon Park",
            "விதான சௌதா": "Vidhana Soudha",
            "இந்திரா நகர்": "Indiranagar",
            "பனசங்கரி": "Banashankari",
            "ஜெயநகர்": "Jayanagar",
            "வைட்ஃபீல்ட்": "Whitefield",
            "மராட்டஹள்ளி": "Marathalli",
            
            # Telugu transliterations
            "మెజెస్టిక్": "Majestic",
            "ఎం జి రోడ్": "MG Road",
            "కబ్బన్ పార్క్": "Cubbon Park",
            "విధాన సౌధ": "Vidhana Soudha",
            "ఇందిరానగర్": "Indiranagar",
            "బనశంకరీ": "Banashankari",
            "జయనగర్": "Jayanagar",
            "వైట్‌ఫీల్డ్": "Whitefield",
            "మరాఠహళ్ళి": "Marathalli"
        }
        
        # Language detection patterns
        self.language_patterns = {
            'en': [
                # Common English words for metro booking
                r'\b(book|ticket|travel|from|to|station|metro|train|price|cost|fare|help)\b',
                r'\b(majestic|indiranagar|whitefield|airport|electronic)\b',
                r'\b(what|how|much|need|want|get|buy)\b'
            ],
            'hi': [
                # Hindi Devanagari script patterns
                r'[ऀ-ॿ]',  # Hindi Unicode range
                r'\b(टिकट|बुक|यात्रा|से|तक|स्टेशन|मेट्रो|कितना|दाम|मदद)\b',
                r'\b(मैजेस्टिक|इंदिरानगर|एमजी|रोड)\b'
            ],
            'mr': [
                # Marathi patterns (shares Devanagari with Hindi but has distinct words)
                r'\b(तिकीट|बुक|प्रवास|पासून|पर्यंत|स्टेशन|मेट्रो|किती|पैसे|मदत)\b',
                r'\b(मेजेस्टिक|इंदिरानगर|एमजी|रोड|व्हाईटफील्ड)\b',
                r'\b(करा|आहे|हवे|पाहिजे|ला|माहिती)\b'
            ],
            'kn': [
                # Kannada script patterns
                r'[ಅ-ೌ]',  # Kannada Unicode range
                r'\b(ಟಿಕೆಟ್|ಬುಕ್|ಪ್ರಯಾಣ|ನಿಂದ|ಗೆ|ಸ್ಟೇಶನ್|ಮೆಟ್ರೋ|ಎಷ್ಟು|ಬೆಲೆ|ಸಹಾಯ)\b',
                r'\b(ಮೆಜೆಸ್ಟಿಕ್|ಇಂದಿರಾನಗರ|ಎಂ|ಜಿ|ರೋಡ್)\b'
            ],
            'ta': [
                # Tamil script patterns
                r'[அ-ௌ]',  # Tamil Unicode range
                r'\b(டிக்கெட்|புக்|பயணம்|இலிருந்து|வரை|ஸ்டேஷன்|மெட்ரோ|எவ்வளவு|விலை|உதவி)\b',
                r'\b(மாஜஸ்டிக்|இந்திரா|நகர்|எம்|ஜி|ரோட்)\b'
            ],
            'te': [
                # Telugu script patterns
                r'[అ-ౌ]',  # Telugu Unicode range
                r'\b(టిక్కెట్|బుక్|ప్రయాణం|నుండి|వరకు|స్టేషన్|మెట్రో|ఎంత|ధర|సహాయం)\b',
                r'\b(మెజెస్టిక్|ఇందిరానగర్|ఎం|జి|రోడ్)\b'
            ]
        }
        
        # Language confidence thresholds
        self.language_confidence_threshold = 0.3
        
    def detect_language(self, text: str) -> str:
        """Automatically detect language from text"""
        if not text:
            return 'en'
            
        text_lower = text.lower()
        
        # Simple script-based detection (most reliable)
        if re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
            return 'kn'
        elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
            return 'ta'
        elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
            return 'te'
        elif re.search(r'[\u0900-\u097F]', text):  # Devanagari
            # Check for Marathi vs Hindi keywords
            marathi_words = ['पासून', 'पर्यंत', 'तिकीट', 'किती', 'मदत', 'माहिती', 'जाणे']
            hindi_words = ['से', 'तक', 'टिकट', 'कितना', 'मदद', 'जानकारी', 'जाना']
            
            marathi_count = sum(1 for word in marathi_words if word in text)
            hindi_count = sum(1 for word in hindi_words if word in text)
            
            return 'mr' if marathi_count > hindi_count else 'hi'
        else:
            # For Roman script text, use vocabulary-based detection
            
            # Hindi romanized keywords
            hindi_roman = ['se', 'tak', 'tikat', 'kitna', 'paisa', 'rupya', 'madad', 'jankari', 'jana', 'karne', 'chahiye', 'kar', 'karo']
            
            # Marathi romanized keywords  
            marathi_roman = ['pasun', 'paryant', 'tikit', 'kiti', 'paise', 'rupye', 'madad', 'mahiti', 'jane', 'karnya', 'pahije', 'kar', 'kara']
            
            # Kannada romanized keywords
            kannada_roman = ['ticket', 'eshtu', 'bele', 'sahaya', 'mahiti', 'hogbeku', 'madi']
            
            # Tamil romanized keywords
            tamil_roman = ['ticket', 'evvalavu', 'vilai', 'udavi', 'thakaval', 'poganum', 'seyya']
            
            # Telugu romanized keywords
            telugu_roman = ['ticket', 'enta', 'dhara', 'sahayam', 'samacharam', 'vellali', 'cheya']
            
            # English keywords
            english_words = ['book', 'ticket', 'travel', 'from', 'to', 'station', 'metro', 'train', 
                           'price', 'cost', 'fare', 'help', 'information', 'go', 'need', 'want']
            
            # Count matches for each language
            scores = {
                'hi': sum(1 for word in hindi_roman if word in text_lower),
                'mr': sum(1 for word in marathi_roman if word in text_lower),
                'kn': sum(1 for word in kannada_roman if word in text_lower),
                'ta': sum(1 for word in tamil_roman if word in text_lower),
                'te': sum(1 for word in telugu_roman if word in text_lower),
                'en': sum(1 for word in english_words if word in text_lower)
            }
            
            # Find the language with highest score
            if max(scores.values()) > 0:
                detected_lang = max(scores, key=scores.get)
                print(f"📊 Vocabulary scores: {scores}")
                print(f"🎯 Best match: {detected_lang}")
                return detected_lang
            else:
                # Fallback: check for common Indian metro station names
                indian_stations = ['majestic', 'indiranagar', 'banashankari', 'jayanagar', 
                                 'whitefield', 'electronic city', 'mg road', 'cubbon park']
                
                if any(station in text_lower for station in indian_stations):
                    print("🚉 Detected Indian station names, defaulting to Hindi")
                    return 'hi'  # Default to Hindi for Indian context
                else:
                    print("🌍 No specific patterns found, defaulting to English")
                    return 'en'
        
    def transliterate_to_native_script(self, text: str, language: str) -> str:
        """
        Transliterate romanized text to native script for supported languages
        
        Args:
            text: Input text (romanized)
            language: Target language code
            
        Returns:
            Text in native script if possible, otherwise original text
        """
        if not TRANSLITERATION_AVAILABLE:
            return text
            
        # Only transliterate if text appears to be romanized (no native script)
        if language == 'hi':
            if not re.search(r'[\u0900-\u097F]', text):  # No Devanagari
                try:
                    return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                except:
                    # Fallback simple mapping for common words
                    return self._simple_hindi_transliteration(text)
        elif language == 'mr':
            if not re.search(r'[\u0900-\u097F]', text):  # No Devanagari
                try:
                    return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                except:
                    return self._simple_marathi_transliteration(text)
        elif language == 'kn':
            if not re.search(r'[\u0C80-\u0CFF]', text):  # No Kannada
                try:
                    return transliterate(text, sanscript.ITRANS, sanscript.KANNADA)
                except:
                    return self._simple_kannada_transliteration(text)
        elif language == 'ta':
            if not re.search(r'[\u0B80-\u0BFF]', text):  # No Tamil
                try:
                    return transliterate(text, sanscript.ITRANS, sanscript.TAMIL)
                except:
                    return self._simple_tamil_transliteration(text)
        elif language == 'te':
            if not re.search(r'[\u0C00-\u0C7F]', text):  # No Telugu
                try:
                    return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
                except:
                    return self._simple_telugu_transliteration(text)
                    
        return text  # Return original if already in native script or unsupported
        
    def _simple_hindi_transliteration(self, text: str) -> str:
        """Simple fallback transliteration for Hindi"""
        replacements = {
            # Station names
            'majestic': 'मैजेस्टिक',
            'indiranagar': 'इंदिरानगर',
            'mg road': 'एमजी रोड',
            'whitefield': 'व्हाइटफील्ड',
            'airport': 'एयरपोर्ट',
            'electronic city': 'इलेक्ट्रॉनिक सिटी',
            'banashankari': 'बनशंकरी',
            'jayanagar': 'जयनगर',
            'koramangala': 'कोरमंगला',
            'marathalli': 'मराठहल्ली',
            
            # Common words  
            'book': 'बुक',
            'ticket': 'टिकट',
            'tickets': 'टिकट',
            'se': 'से', 
            'from': 'से',
            'tak': 'तक',
            'to': 'तक',
            'kitna': 'कितना',
            'paisa': 'पैसा',
            'kar': 'कर',
            'karo': 'करो',
            'chahiye': 'चाहिए',
            'jana': 'जाना',
            'jaana': 'जाना',
            'help': 'मदद',
            'madad': 'मदद',
            'can you': 'क्या आप',
            'please': 'कृपया',
            'hey': 'अरे',
            'there': 'वहाँ'
        }
        result = text.lower()
        for eng, hindi in replacements.items():
            result = result.replace(eng, hindi)
        return result
        
    def _simple_marathi_transliteration(self, text: str) -> str:
        """Simple fallback transliteration for Marathi"""
        replacements = {
            # Station names
            'majestic': 'मेजेस्टिक',
            'indiranagar': 'इंदिरानगर',
            'mg road': 'एमजी रोड',
            'whitefield': 'व्हाईटफील्ड',
            'airport': 'एअरपोर्ट',
            'electronic city': 'इलेक्ट्रॉनिक शहर',
            'banashankari': 'बानशंकरी',
            'jayanagar': 'जयनगर',
            'koramangala': 'कोरमंगला',
            'marathalli': 'मराठाहल्ली',
            
            # Common words
            'book': 'बुक',
            'ticket': 'तिकीट',
            'tickets': 'तिकीट',
            'pasun': 'पासून',
            'from': 'पासून',
            'paryant': 'पर्यंत',
            'to': 'पर्यंत',
            'kiti': 'किती',
            'paise': 'पैसे',
            'kar': 'कर',
            'kara': 'करा',
            'pahije': 'पाहिजे',
            'jane': 'जाणे',
            'jaane': 'जाणे',
            'help': 'मदत',
            'madad': 'मदत',
            'can you': 'तुम्ही',
            'please': 'कृपया',
            'hey': 'अरे',
            'there': 'तिथे'
        }
        result = text.lower()
        for eng, marathi in replacements.items():
            result = result.replace(eng, marathi)
        return result
        
    def _simple_kannada_transliteration(self, text: str) -> str:
        """Simple fallback transliteration for Kannada"""
        replacements = {
            'majestic': 'ಮೆಜೆಸ್ಟಿಕ್',
            'indiranagar': 'ಇಂದಿರಾನಗರ',
            'mg road': 'ಎಂ ಜಿ ರೋಡ್',
            'whitefield': 'ವೈಟ್‌ಫೀಲ್ಡ್',
            'book': 'ಬುಕ್',
            'ticket': 'ಟಿಕೆಟ್'
        }
        result = text.lower()
        for eng, kannada in replacements.items():
            result = result.replace(eng, kannada)
        return result
        
    def _simple_tamil_transliteration(self, text: str) -> str:
        """Simple fallback transliteration for Tamil"""
        replacements = {
            'majestic': 'மாஜஸ்டிக்',
            'indiranagar': 'இந்திரா நகர்',
            'mg road': 'எம் ஜி ரோட்',
            'whitefield': 'வைட்ஃபீல்ட்',
            'book': 'புக்',
            'ticket': 'டிக்கெட்'
        }
        result = text.lower()
        for eng, tamil in replacements.items():
            result = result.replace(eng, tamil)
        return result
        
    def _simple_telugu_transliteration(self, text: str) -> str:
        """Simple fallback transliteration for Telugu"""
        replacements = {
            'majestic': 'మెజెస్టిక్',
            'indiranagar': 'ఇందిరానగర్',
            'mg road': 'ఎం జి రోడ్',
            'whitefield': 'వైట్‌ఫీల్డ్',
            'book': 'బుక్',
            'ticket': 'టిక్కెట్'
        }
        result = text.lower()
        for eng, telugu in replacements.items():
            result = result.replace(eng, telugu)
        return result
        
    def process_text(self, text: str, language: str = None) -> Dict[str, Any]:
        """
        Process text for intent detection and entity extraction
        
        Args:
            text: Input text to process
            language: Language code (en, hi, kn, ta, te, mr) - auto-detected if None
            
        Returns:
            Dictionary with intent, entities, and extracted information
        """
        try:
            # Auto-detect language if not provided
            if language is None:
                language = self.detect_language(text)
                print(f"🔍 Auto-detected language: {language}")
            
            print(f"🧠 Processing: '{text}' in {language}")
            
            # Normalize text
            normalized_text = self._normalize_text(text, language)
            print(f"📝 Normalized: '{normalized_text}'")
            
            # Process with enhanced rules
            return self._process_with_rules(normalized_text, language)
            
        except Exception as e:
            print(f"❌ Language processing error: {e}")
            return self._get_default_response(text, language)
    
    def _normalize_text(self, text: str, language: str) -> str:
        """Normalize text for processing while preserving Indian language characters"""
        # Convert to lowercase for processing
        text = text.lower().strip()
        
        # For Indian languages, preserve special characters
        if language in ['hi', 'mr', 'kn', 'ta', 'te']:
            # Only remove basic punctuation, keep Devanagari and other scripts
            text = re.sub(r'[.,!?;:]', ' ', text)
        else:
            # For English, remove punctuation except essential ones
            text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _process_with_rules(self, text: str, language: str) -> Dict[str, Any]:
        """Enhanced rule-based processing for intent detection with multiple intents support"""
        print(f"🔍 Processing text: '{text}'")
        print(f"🌍 Language: {language}")
        
        result = {
            'intent': 'unknown',
            'confidence': 0.0,
            'entities': {},
            'text': text,
            'language': language
        }
        
        # 1. CANCEL TICKET INTENT (Highest Priority)
        cancel_patterns = [
            # English patterns
            r'cancel.*ticket', r'ticket.*cancel', r'refund', r'cancellation',
            r'cancel.*booking', r'booking.*cancel', r'return.*ticket',
            
            # Hindi patterns (Devanagari)
            r'टिकट.*रद्द', r'रद्द.*टिकट', r'टिकट.*वापस', r'वापस.*टिकट',
            r'बुकिंग.*रद्द', r'रद्द.*बुकिंग', r'रिफंड', r'कैंसल',
            r'रद्द.*करो', r'कैंसल.*करो', r'टिकट.*कैंसल', r'कैंसल.*टिकट',
            
            # Hindi patterns (Romanized)  
            r'tikat.*cancel', r'cancel.*tikat', r'tikat.*radd', r'radd.*tikat',
            r'cancel.*karo', r'radd.*karo', r'tikat.*wapas', r'booking.*cancel',
            
            # Marathi patterns (Devanagari) 
            r'तिकीट.*रद्द', r'रद्द.*तिकीट', r'तिकीट.*परत', r'परत.*तिकीट',
            r'बुकिंग.*रद्द', r'रद्द.*बुकिंग', r'रिफंड', r'कॅन्सल',
            r'रद्द.*करा', r'कॅन्सल.*करा', r'तिकीट.*कॅन्सल', r'कॅन्सल.*तिकीट',
            
            # Marathi patterns (Romanized)
            r'tikit.*cancel', r'cancel.*tikit', r'tikit.*radd', r'radd.*tikit',
            r'cancel.*kara', r'radd.*kara', r'tikit.*parat', r'booking.*cancel',
            
            # Kannada patterns
            r'ಟಿಕೆಟ್.*ರದ್ದು', r'ರದ್ದು.*ಟಿಕೆಟ್', r'ಕ್ಯಾನ್ಸಲ್',
            
            # Tamil patterns
            r'டிக்கெட்.*ரத்து', r'ரத்து.*டிக்கெட்', r'கேன்சல்',
            
            # Telugu patterns
            r'టిక్కెట్.*రద్దు', r'రద్దు.*టిక్కెట్', r'క్యాన్సల్'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in cancel_patterns):
            result['intent'] = 'cancel_ticket'
            result['confidence'] = 0.9
            print("✅ Cancel ticket intent detected")
            
            # Try to extract booking ID or stations
            booking_id_match = re.search(r'(BM[A-Z0-9]{8}|[A-Z0-9]{8,12})', text)
            if booking_id_match:
                result['entities']['booking_id'] = booking_id_match.group(1)
                result['booking_id'] = booking_id_match.group(1)
                
            stations = self._extract_stations(text)
            if stations:
                result['entities']['stations'] = stations
                
            return result
        
        # 2. FARE/PRICE INQUIRY INTENT (High Priority)
        price_patterns = [
            # English patterns
            r'price', r'cost', r'fare', r'how much', r'charges', r'rate',
            r'what.*cost', r'what.*price', r'cost.*travel', r'fare.*from',
            
            # Hindi patterns (Devanagari)
            r'कितना', r'दाम', r'कीमत', r'किमत', r'फेयर', r'पैसा', r'रुपया',
            r'क्या.*दाम', r'क्या.*कीमत', r'कितने.*पैसे', r'कितना.*खर्च',
            r'कितना.*पैसा.*लगेगा', r'कितना.*पैसा.*लगता', r'कितना.*लगेगा',
            r'पैसा.*लगेगा', r'खर्च.*कितना', r'दाम.*क्या',
            
            # Hindi patterns (Romanized)
            r'kitna', r'daam', r'keemat', r'kaimat', r'paisa', r'rupya',
            r'kitna.*paisa', r'paisa.*kitna', r'kitna.*lagega', r'paisa.*lagega',
            r'kitne.*paise', r'kitna.*kharcha', r'kitna.*cost',
            
            # Marathi patterns (Devanagari)
            r'किती', r'दर', r'किंमत', r'फेअर', r'पैसे', r'रुपये',
            r'ऐंकडे', r'खर्च', r'भाडे', r'दर', r'शुल्क',
            r'काय.*दर', r'काय.*किंमत', r'किती.*पैसे', r'किती.*खर्च',
            r'किती.*पैसे.*लागतील', r'किती.*खर्च', r'पैसे.*किती',
            
            # Marathi patterns (Romanized)
            r'kiti', r'dar', r'kinmat', r'paise', r'rupye',
            r'kiti.*paise', r'paise.*kiti', r'kiti.*kharcha', r'kiti.*lagel',
            
            # Kannada patterns
            r'ಎಷ್ಟು', r'ಬೆಲೆ', r'ದರ', r'ಫೇರ್', r'ಎಷ್ಟು.*ಬೆಲೆ',
            
            # Tamil patterns
            r'எவ்வளவு', r'விலை', r'கட்டணம்', r'ஃபேர்', r'எவ்வளவு.*விலை',
            
            # Telugu patterns
            r'ఎంత', r'ధర', r'రేటు', r'ఫేర్', r'ఎంత.*ధర'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in price_patterns):
            result['intent'] = 'fare_inquiry'
            result['confidence'] = 0.9
            print("✅ Fare inquiry intent detected")
            
            stations = self._extract_stations(text)
            if len(stations) >= 2:
                result['entities']['from_station'] = stations[0]
                result['entities']['to_station'] = stations[1]
                result['from_station'] = stations[0]
                result['to_station'] = stations[1]
                result['confidence'] = 0.95
            elif len(stations) == 1:
                result['entities']['station'] = stations[0]
                
            # Extract quantity for fare calculation
            quantity = self._extract_quantity(text)
            if quantity > 1:
                result['entities']['quantity'] = quantity
                result['quantity'] = quantity
                
            return result
        
        # 3. STATION INFO/ROUTE INQUIRY INTENT
        route_patterns = [
            # English patterns
            r'route.*to', r'how.*get.*to', r'way.*to', r'direction.*to',
            r'metro.*route', r'stations.*between', r'path.*to', r'travel.*route',
            
            # Hindi patterns (Devanagari)
            r'रास्ता', r'मार्ग', r'कैसे.*जाएं', r'कैसे.*पहुंचे', r'दिशा',
            r'स्टेशन.*बीच', r'रूट.*तक', r'मेट्रो.*मार्ग',
            r'कैसे.*जाये', r'कैसे.*जाना', r'कैसे.*पहुंचे', r'जाने.*का.*रास्ता',
            
            # Hindi patterns (Romanized)
            r'rasta', r'marg', r'kaise.*jaaye', r'kaise.*pohoche', r'disha',
            r'kaise.*jana', r'kaise.*jaye', r'jane.*ka.*rasta', r'route.*kaise',
            
            # Marathi patterns (Devanagari)
            r'रस्ता', r'मार्ग', r'कसे.*जायचे', r'कसे.*पोहोचायचे', r'दिशा',
            r'स्टेशन.*मधील', r'रूट.*पर्यंत', r'मेट्रो.*मार्ग',
            r'कसे.*जायचे', r'कसे.*जाणे', r'जाण्याचा.*मार्ग',
            
            # Marathi patterns (Romanized)
            r'rasta', r'marg', r'kase.*jayche', r'kase.*pohochayche', r'disha',
            r'kase.*jane', r'kase.*jayche', r'jane.*cha.*marg', r'route.*kase',
            
            # Kannada patterns
            r'ರೂಟ್', r'ಮಾರ್ಗ', r'ಹೇಗೆ.*ಹೋಗಬೇಕು', r'ದಿಕ್ಕು',
            
            # Tamil patterns
            r'ரூட்', r'வழி', r'எப்படி.*செல்வது', r'திசை',
            
            # Telugu patterns
            r'రూట్', r'మార్గం', r'ఎలా.*వెళ్ళాలి', r'దిశ'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in route_patterns):
            result['intent'] = 'route_inquiry'
            result['confidence'] = 0.85
            print("✅ Route inquiry intent detected")
            
            stations = self._extract_stations(text)
            if stations:
                result['entities']['stations'] = stations
                if len(stations) >= 2:
                    result['entities']['from_station'] = stations[0]
                    result['entities']['to_station'] = stations[1]
                    result['from_station'] = stations[0]
                    result['to_station'] = stations[1]
                    
            return result
        
        # 4. BOOKING STATUS/HISTORY INQUIRY
        status_patterns = [
            # English patterns
            r'booking.*status', r'status.*booking', r'my.*booking', r'booking.*history',
            r'check.*booking', r'booking.*details', r'ticket.*status', r'my.*ticket',
            
            # Hindi patterns
            r'बुकिंग.*स्थिति', r'स्थिति.*बुकिंग', r'मेरी.*बुकिंग', r'बुकिंग.*इतिहास',
            r'टिकट.*स्थिति', r'मेरा.*टिकट', r'चेक.*बुकिंग',
            
            # Marathi patterns
            r'बुकिंग.*स्थिती', r'स्थिती.*बुकिंग', r'माझी.*बुकिंग', r'बुकिंग.*इतिहास',
            r'तिकीट.*स्थिती', r'माझे.*तिकीट', r'चेक.*बुकिंग',
            
            # Kannada patterns
            r'ಬುಕಿಂಗ್.*ಸ್ಥಿತಿ', r'ನನ್ನ.*ಬುಕಿಂಗ್', r'ಟಿಕೆಟ್.*ಸ್ಥಿತಿ',
            
            # Tamil patterns
            r'புக்கிங்.*நிலை', r'என்.*புக்கிங்', r'டிக்கெட்.*நிலை',
            
            # Telugu patterns
            r'బుకింగ్.*స్థితి', r'నా.*బుకింగ్', r'టిక్కెట్.*స్థితి'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in status_patterns):
            result['intent'] = 'booking_status'
            result['confidence'] = 0.85
            print("✅ Booking status inquiry intent detected")
            
            # Try to extract booking ID
            booking_id_match = re.search(r'(BM[A-Z0-9]{8}|[A-Z0-9]{8,12})', text)
            if booking_id_match:
                result['entities']['booking_id'] = booking_id_match.group(1)
                result['booking_id'] = booking_id_match.group(1)
                
            return result
        
        # 5. GENERAL INQUIRY/HELP INTENT
        help_patterns = [
            # English patterns
            r'help.*metro', r'metro.*help', r'metro.*information', r'information.*metro',
            r'help.*information', r'information.*help', r'general.*help', r'metro.*details',
            r'metro.*timings', r'metro.*schedule', r'metro.*route.*info',
            
            # Hindi patterns
            r'मदद.*मेट्रो', r'मेट्रो.*मदद', r'मेट्रो.*जानकारी', r'जानकारी.*मेट्रो',
            r'मदद.*जानकारी', r'जानकारी.*मदद', r'मेट्रो.*सहायता', r'सहायता.*मेट्रो',
            r'मेट्रो.*समय', r'मेट्रो.*शेड्यूल',
            
            # Marathi patterns
            r'मदत.*मेट्रो', r'मेट्रो.*मदत', r'मेट्रो.*माहिती', r'माहिती.*मेट्रो',
            r'मदत.*माहिती', r'माहिती.*मदत', r'मेट्रो.*सहाय्य', r'सहाय्य.*मेट्रो',
            r'मेट्रो.*वेळ', r'मेट्रो.*वेळापत्रक',
            
            # Kannada patterns
            r'ಸಹಾಯ.*ಮೆಟ್ರೋ', r'ಮೆಟ್ರೋ.*ಸಹಾಯ', r'ಮೆಟ್ರೋ.*ಮಾಹಿತಿ',
            
            # Tamil patterns  
            r'உதவி.*மெட்ரோ', r'மெட்ரோ.*உதவி', r'மெட்ரோ.*தகவல்',
            
            # Telugu patterns
            r'సహాయం.*మెట్రో', r'మెట్రో.*సహాయం', r'మెట్రో.*సమాచారం'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in help_patterns):
            result['intent'] = 'general_inquiry'
            result['confidence'] = 0.85
            print("✅ General inquiry intent detected")
            return result
        
        # 6. ENHANCED BOOKING INTENT (Last priority)
        booking_patterns = [
            # English patterns
            r'book.*ticket', r'ticket.*book', r'want.*travel', r'need.*ticket',
            r'go.*from', r'travel.*to', r'journey.*from', r'trip.*to',
            r'from.*to', r'get.*ticket', r'buy.*ticket', r'ticket.*from',
            r'travel', r'book', r'ticket', r'metro', r'train',
            
            # Simple station patterns (A to B, A se B)
            r'\w+\s+(to|tak|paryant|ge|varaku)\s+\w+',
            r'\w+\s+(se|pasun|ninda|to)\s+\w+',
            
            # Hindi patterns
            r'टिकट.*बुक', r'बुक.*टिकट', r'यात्रा.*करना', r'जाना.*है',
            r'से.*के लिए', r'से.*तक', r'के लिए', r'टिकट.*चाहिए',
            r'टिकट', r'यात्रा', r'बुक', r'जाना',
            
            # Marathi patterns (enhanced)
            r'तिकीट.*बुक', r'बुक.*तिकीट', r'तिकिट.*बुक', r'बुक.*तिकिट',
            r'प्रवास.*करायचा', r'प्रवास.*करणे', r'जायचे.*आहे', r'जाणे.*आहे',
            r'पासून.*पर्यंत', r'पासून.*ला', r'कडून.*पर्यंत', r'ला.*जायचे',
            r'तिकीट.*हवे', r'तिकीट.*पाहिजे', r'तिकिट.*हवे', r'तिकिट.*पाहिजे',
            r'तिकीट', r'तिकिट', r'प्रवास', r'बुक', r'जाणे', r'यात्रा',
            
            # Kannada patterns
            r'ಟಿಕೆಟ್.*ಬುಕ್', r'ಬುಕ್.*ಟಿಕೆಟ್', r'ಹೋಗಬೇಕು', r'ಪ್ರಯಾಣ',
            r'ನಿಂದ.*ಗೆ', r'ಟಿಕೆಟ್.*ಬೇಕು', r'ಟಿಕೆಟ್', r'ಪ್ರಯಾಣ', r'ಬುಕ್',
            
            # Tamil patterns
            r'டிக்கெட்.*புக்', r'புக்.*டிக்கெட்', r'செல்ல.*வேண்டும்', r'பயணம்',
            r'இலிருந்து.*வரை', r'டிக்கெட்.*வேண்டும்', r'டிக்கெட்', r'பயணம்', r'புக்',
            
            # Telugu patterns
            r'టిక్కెట్.*బుక్', r'బుక్.*టిక్కెట్', r'వెళ్ళాలి', r'ప్రయాణం',
            r'నుండి.*వరకు', r'టిక్కెట్.*కావాలి', r'టిక్కెట్', r'ప్రయాణం', r'బుక్'
        ]
        
        # Check each pattern and log matches
        matched_patterns = []
        for pattern in booking_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        print(f"📝 Matched patterns: {matched_patterns}")
        
        if matched_patterns:
            result['intent'] = 'book_ticket'
            result['confidence'] = 0.8
            print(f"✅ Booking intent detected with {len(matched_patterns)} pattern matches")
            
            # Extract stations
            stations = self._extract_stations(text)
            print(f"🚉 Extracted stations: {stations}")
            
            if len(stations) >= 2:
                result['entities']['from_station'] = stations[0]
                result['entities']['to_station'] = stations[1]
                result['from_station'] = stations[0]
                result['to_station'] = stations[1]
                result['confidence'] = 0.9
            elif len(stations) == 1:
                result['entities']['station'] = stations[0]
                result['confidence'] = 0.6
            
            # Extract quantity
            quantity = self._extract_quantity(text)
            result['quantity'] = quantity
            if quantity > 1:
                result['entities']['quantity'] = quantity
        
        print(f"🎯 Final result: {result['intent']} ({result['confidence']})")
        return result
    
    def _extract_stations(self, text: str) -> List[str]:
        """Extract metro station names from text with enhanced transliteration support"""
        print(f"🔍 Extracting stations from: '{text}'")
        found_stations = []
        text_lower = text.lower()
        
        # Check for exact matches first
        for station in self.metro_stations:
            if station.lower() in text_lower and station not in found_stations:
                found_stations.append(station)
                print(f"✅ Found exact match: {station}")
        
        # Check aliases (including transliterations)
        for alias, station in self.station_aliases.items():
            if alias in text_lower and station not in found_stations:
                found_stations.append(station)
                print(f"✅ Found alias/transliteration match: {alias} -> {station}")
        
        # Fuzzy matching for partial names (more restrictive)
        words = text.split()
        for word in words:
            if len(word) > 4:  # Only check words longer than 4 chars
                # Skip common words that aren't station names
                skip_words = ['help', 'information', 'metro', 'ticket', 'book', 'travel', 
                             'jankari', 'chahiye', 'karo', 'kara', 'lagega', 'paisa', 'se', 'tak']
                if word.lower() in skip_words:
                    continue
                    
                matches = get_close_matches(word, [s.lower() for s in self.metro_stations], n=1, cutoff=0.7)
                if matches:
                    station = next(s for s in self.metro_stations if s.lower() == matches[0])
                    if station not in found_stations:
                        found_stations.append(station)
                        print(f"✅ Found fuzzy match: {word} -> {station}")
        
        print(f"🚉 Total stations found: {found_stations}")
        return found_stations
    
    def _extract_quantity(self, text: str) -> int:
        """Extract ticket quantity from text"""
        # Number words mapping for multiple languages
        number_words = {
            # English
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            
            # Hindi (Devanagari)
            'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5, 'पाँच': 5,
            'छह': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
            
            # Hindi (Romanized)
            'ek': 1, 'do': 2, 'teen': 3, 'char': 4, 'panch': 5, 'paanch': 5,
            'chah': 6, 'saat': 7, 'aath': 8, 'nau': 9, 'das': 10,
            
            # Marathi (Devanagari)
            'दोन': 2, 'पाच': 5, 'सहा': 6, 'नऊ': 9, 'दहा': 10,
            'एका': 1, 'दोघा': 2, 'तिघा': 3, 'चौघा': 4, 'पाचजण': 5,
            
            # Marathi (Romanized)
            'don': 2, 'panch': 5, 'saha': 6, 'nau': 9, 'daha': 10,
            'doghi': 2, 'tighi': 3, 'choghi': 4, 'pachjan': 5,
            
            # Kannada
            'ಒಂದು': 1, 'ಎರಡು': 2, 'ಮೂರು': 3, 'ನಾಲ್ಕು': 4, 'ಐದು': 5,
            'ಆರು': 6, 'ಏಳು': 7, 'ಎಂಟು': 8, 'ಒಂಬತ್ತು': 9, 'ಹತ್ತು': 10,
            
            # Tamil
            'ஒன்று': 1, 'இரண்டு': 2, 'மூன்று': 3, 'நான்கு': 4, 'ஐந்து': 5,
            'ஆறு': 6, 'ஏழு': 7, 'எட்டு': 8, 'ஒன்பது': 9, 'பத்து': 10,
            
            # Telugu
            'ఒకటి': 1, 'రెండు': 2, 'మూడు': 3, 'నాలుగు': 4, 'ఐదు': 5,
            'ఆరు': 6, 'ఏడు': 7, 'ఎనిమిది': 8, 'తొమ్మిది': 9, 'పది': 10
        }
        
        # Look for digit numbers first (most reliable)
        numbers = re.findall(r'\d+', text)
        if numbers:
            quantity = int(numbers[0])
            print(f"🔢 Found digit: {quantity}")
            return min(quantity, 10)  # Cap at 10 tickets
        
        # Look for quantity patterns like "for X people", "X passengers", etc.
        people_patterns = [
            r'(\d+)\s+(?:people|passengers|persons|लोग|लोगों|व्यक्ति|जन)',
            r'for\s+(\d+)',
            r'(\d+)\s+(?:tickets?|टिकट|तिकीट|ಟಿಕೆಟ್|டிக்கெட்|టిక్కెట్)'
        ]
        
        for pattern in people_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                quantity = int(match.group(1))
                print(f"🔢 Found people/ticket pattern: {quantity}")
                return min(quantity, 10)
        
        # Look for word numbers (exact word matches only)
        text_words = text.lower().split()
        for word in text_words:
            if word in number_words:
                print(f"🔢 Found word number: {word} = {number_words[word]}")
                return number_words[word]
        
        print("🔢 No quantity found, defaulting to 1")
        return 1  # Default to 1 ticket
    
    def _get_default_response(self, text: str, language: str) -> Dict[str, Any]:
        """Get default response when processing fails"""
        return {
            'intent': 'unknown',
            'confidence': 0.1,
            'entities': {},
            'text': text,
            'language': language,
            'message': 'Could not understand the request. Please try again.'
        }
