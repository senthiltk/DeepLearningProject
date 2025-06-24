# Metro Booking Voice Assistant

This project implements a voice-driven metro ticket booking system that allows users to send a single voice message containing all the necessary booking details. The system automatically detects the language, transcribes the speech, extracts the user's intent, and proceeds to the booking step.

## Features

- **Language Identification**: Detect the language of the spoken input (English, Hindi, Tamil, Telugu, Kannada, and Malayalam)
- **Speech Recognition**: Transcribe the spoken input into text using language-specific models
- **Intent Detection**: Understand the user's booking intent and extract relevant details
- **Integration**: Connect with the metro ticketing system (simulated)

## Project Structure

```
Project/
├── main.py                  # Main entry point for the application
├── requirements.txt         # Dependencies
├── data/                    # Directory for data files
├── models/                  # Directory for saved models
├── modules/                 # Core modules
│   ├── __init__.py          # Package initialization
│   ├── language_identification.py  # Language identification module
│   ├── speech_recognition.py      # Speech recognition module
│   └── intent_detection.py        # Intent detection module
├── tests/                   # Test cases
│   ├── test_language_identification.py
│   ├── test_speech_recognition.py
│   ├── test_intent_detection.py
│   ├── test_audio_utils.py
│   ├── test_integration.py
│   └── run_tests.py         # Test runner script
└── utils/                   # Utility functions
    ├── __init__.py          # Package initialization
    └── audio_utils.py       # Audio processing utilities
```

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. **macOS Users**: Fix SSL certificate issues by running:

```bash
./fix_ssl_certificates.sh
```

4. **FFmpeg Installation**: This project requires FFmpeg for audio processing. Install it if not already present:

- **macOS**:
  ```bash
  brew install ffmpeg
  ```
  
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install ffmpeg
  ```
  
- **Windows**: Download and install from [FFmpeg Official Website](https://ffmpeg.org/download.html)

## Usage

### Process a voice message

```bash
python main.py --audio path/to/audio_file.wav
```

The script automatically converts non-WAV audio files (like .m4a, .mp3) to WAV format for better compatibility.

Additional options:
- `--output`: Path to save the booking details (default: booking_result.json)
- `--device`: Device to run models on (cuda/cpu)

### Troubleshooting

If you encounter SSL certificate issues (especially on macOS), run:

```bash
./fix_ssl_certificates.sh
```

If you get an FFmpeg not found error, ensure it's installed and in your PATH:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Add FFmpeg to your system PATH

## Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py -m test_language_identification

# Run with less verbose output
python tests/run_tests.py -q
```

## Components

### 1. Language Identification Module

Uses a pre-trained Whisper model to identify the language of the spoken input. Supports:
- English
- Hindi
- Tamil
- Telugu
- Kannada
- Malayalam

### 2. Speech Recognition Module

Transcribes the spoken input into text using language-specific models:
- For English: OpenAI Whisper English-specific model
- For Indian languages: OpenAI Whisper multilingual model or AI4Bharat IndicASR models

### 3. Intent Detection Module

Extracts booking details from the transcribed text:
- Source station
- Destination station
- Number of tickets

## License

This project is licensed under the MIT License - see the LICENSE file for details.
