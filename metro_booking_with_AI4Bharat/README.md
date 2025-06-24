# Bangalore Metro Voice Booking System

A simplified voice-enabled ticket booking system for Bangalore Metro with multilingual support.

## Features

- **Real Speech Recognition**: Uses Google Speech API for accurate voice-to-text
- **Multilingual Support**: English, Hindi, Kannada, Tamil, Telugu, Marathi
- **Intent Detection**: Smart booking request processing
- **Simple Architecture**: Just 4 core files for easy maintenance

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python3 main.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5002`

4. **Use Voice Booking**:
   - Select your language
   - Click "🎤 Click to Speak"
   - Say: "Book a ticket from Majestic to Indiranagar"
   - View transcription and booking results

## File Structure

```
metro_booking/
├── main.py              # Flask app with embedded UI
├── asr_service.py       # Speech recognition service
├── language_processor.py # Intent detection and NLP
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Example Voice Commands

**English:**
- "Book a ticket from Majestic to Indiranagar"
- "I want to travel from MG Road to Banashankari"
- "Get me two tickets from Whitefield to City Railway Station"
- "How much is a ticket from Vijayanagar to Rajajinagar"

**Hindi:**
- "मैजेस्टिक से एमजी रोड तक टिकट बुक करें"
- "इंदिरानगर से बनशंकरी तक कितना पैसा लगेगा?"

**Marathi:**
- "मेजेस्टिक पासून एमजी रोड पर्यंत तिकीट बुक करा"
- "इंदिरानगर पासून बानशंकरी पर्यंत किती पैसे लागतील?"

**Kannada:**
- "ಮೆಜೆಸ್ಟಿಕ್ ನಿಂದ ಎಂ ಜಿ ರೋಡ್ ಗೆ ಟಿಕೆಟ್ ಬುಕ್ ಮಾಡಿ"

**Tamil:**
- "மாஜஸ்டிக் இலிருந்து எம் ஜி ரோட் வரை டிக்கெட் புக் செய்யுங்கள்"

**Telugu:**
- "మెజెస్టిక్ నుండి ఎం జి రోడ్ వరకు టిక్కెట్ బుక్ చేయండి"

## Technical Details

- **Backend**: Flask with Python 3
- **Speech Recognition**: Google Speech API + SpeechRecognition library
- **NLP**: Rule-based intent detection with fuzzy station matching
- **Frontend**: Embedded HTML/CSS/JavaScript
- **Audio**: Browser-based MediaRecorder API

## Supported Metro Stations

Covers all major Bangalore Metro stations including:
- Majestic, MG Road, Indiranagar, Banashankari
- Whitefield, Electronic City, Airport
- And many more...

## Dependencies

- flask: Web framework
- speechrecognition: Speech-to-text conversion
- librosa: Audio processing
- soundfile: Audio file handling
- requests: HTTP requests
- numpy: Numerical operations

Simple, clean, and ready to use!
