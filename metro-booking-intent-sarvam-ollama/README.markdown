# Voice-Based Metro Ticket Booking System with Indic Languages

This project is a voice-based metro ticket booking system designed to enhance accessibility for Indian commuters by supporting 11 Indic languages, including Hindi and Tamil. Built as a Streamlit web application, it leverages Sarvam AI's multilingual models for speech-to-text (Saarika), translation (Mayura), and intent recognition (Sarvam-M), integrated with a Retrieval-Augmented Generation (RAG) framework and a local large language model (Ollama). The system enables users to book metro tickets, check balances, or cancel bookings through voice commands, addressing barriers for non-English speakers and low-literacy users.

## Features

- Supports 11 Indic languages (e.g., Hindi, Tamil, Bengali) for voice input and output.
- Real-time speech-to-text transcription using Sarvam AI's Saarika model.
- Multilingual translation with Sarvam AI's Mayura model for English processing.
- Intent recognition (e.g., booking, balance inquiry) using RAG and Ollama's llama3 model.
- User-friendly Streamlit interface for recording audio, selecting languages, and viewing results.

## Technologies Used

- **Programming Language**: Python 3.9
- **Web Framework**: Streamlit
- **Audio Processing**: sounddevice, wavio
- **NLP/ML**: sentence-transformers (all-MiniLM-L6-v2), NumPy
- **APIs**: Sarvam AI (Saarika, Bulbul, Mayura, Sarvam-M)
- **LLM**: Ollama (llama3 model)
- **HTTP Requests**: requests
- **Configuration**: config.py for API keys and RAG knowledge base

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/metro-ticket-booking.git
   cd metro-ticket-booking
   ```
2. **Install Python Dependencies**:
   - Ensure Python 3.9 is installed.
   - Install required packages:

     ```bash
     pip install -r requirements.txt
     ```
   - Sample `requirements.txt`:

     ```
     streamlit==1.25.0
     sounddevice==0.4.6
     wavio==0.0.7
     sentence-transformers==2.2.2
     requests==2.31.0
     numpy==1.24.3
     ```
3. **Install Portaudio** (for sounddevice):
   - On Windows: Download from http://www.portaudio.com/ or use a package manager.
   - On macOS: `brew install portaudio`
   - On Linux: `sudo apt-get install libportaudio2`
4. **Set Up Sarvam AI API Keys**:
   - Obtain API keys from Sarvam AI (https://sarvam.ai).
   - Create or edit `config.py`:

     ```python
     SARVAM_API_KEY = "your-sarvam-api-key"
     ```
5. **Install and Run Ollama**:
   - Download Ollama from https://ollama.ai.
   - Install and pull the llama3 model:

     ```bash
     ollama pull llama3
     ollama run llama3
     ```
6. **Verify Setup**:
   - Ensure all dependencies are installed and Ollama is running.

## Usage

1. **Start the Streamlit App**:

   ```bash
   streamlit run main.py
   ```
   - Open the displayed URL (e.g., http://localhost:8501) in a browser.
2. **Interact with the App**:
   - Select a language (e.g., Hindi) from the sidebar.
   - Choose recording duration (e.g., 5 seconds).
   - Click “Record” to capture voice input (e.g., “Book a ticket from Connaught Place to Karol Bagh”).
   - View transcription, translation, intent, and response in the UI.
   - Hear the response via text-to-speech (Bulbul).
3. **Example Interaction**:
   - Input (Hindi): “Dilli Haat se Rajiv Chowk tak ek ticket book karo.”
   - Output: Transcription → Translation (to English) → Intent (booking) → Response (e.g., “Ticket booked!”) → Spoken response in Hindi.

## Project Structure

- `main.py`: Streamlit app, backend ticketing logic, and UI integration.
- `audio_processing.py`: Handles audio recording and WAV file creation using sounddevice and wavio.
- `sarvam_api.py`: Integrates Sarvam AI APIs for speech-to-text, text-to-speech, and translation.
- `rag_system.py`: Implements RAG for intent recognition using sentence-transformers.
- `llm_interface.py`: Interfaces with Ollama for LLM-based intent classification.
- `config.py`: Stores Sarvam API keys and RAG knowledge base (intent examples).
- `requirements.txt`: Lists Python dependencies.