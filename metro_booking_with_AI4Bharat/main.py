#!/usr/bin/env python3
"""
Bangalore Metro Booking System - Main Entry Point
Simple Flask web application for voice-based metro ticket booking
"""

from flask import Flask, render_template, request, jsonify
import os
import uuid
import tempfile
from asr_service import SpeechRecognition
from language_processor import LanguageProcessor

app = Flask(__name__)
app.secret_key = 'metro-booking-secret-key'

# Initialize services
speech_service = SpeechRecognition()
language_service = LanguageProcessor()

# Metro stations data
METRO_STATIONS = [
    "Majestic", "City Railway Station", "Magadi Road", "Hosahalli",
    "Vijayanagar", "Attiguppe", "Deepanjali Nagar", "Mysore Road",
    "Nayandahalli", "Rajarajeshwari Nagar", "Jnanabharathi",
    "Pattanagere", "Kengeri Bus Terminal", "Kengeri",
    "Challaghatta", "Vajarahalli", "Thalaghattapura", "Silk Institute",
    "Nagasandra", "Dasarahalli", "Jalahalli", "Peenya Industry",
    "Peenya", "Goraguntepalya", "Yeshwantpur", "Sandal Soap Factory",
    "Mahalakshmi", "Rajajinagar", "Mahakavi Kuvempu Road", "Srirampura",
    "Sampige Road", "Nadaprabhu Kempegowda Station", "Chickpet",
    "Krishna Rajendra Market", "National College", "Lalbagh",
    "South End Circle", "Jayanagar", "Rashtreeya Vidyalaya Road",
    "Banashankari", "Jayaprakash Nagar", "Yelachenahalli",
    "Konanakunte Cross", "Doddakallasandra", "Vajarahalli",
    "Thalghattapura", "Silk Institute", "Kanakapura Road",
    "Ittamadu", "Anjanapura", "Jaraganahalli", "Pattandur Agrahara",
    "Banashankari Temple", "Rashtriya Vidyalaya Road", "Jayanagar East",
    "Lalbagh West", "KR Market West", "Chickpet West", "Majestic East",
    "Cantonment", "Shivaji Nagar", "Cubbon Park", "Vidhana Soudha",
    "Sir M Visvesvaraya Station Central College", "Nadaprabhu Kempegowda Station Majestic",
    "City Railway Station Majestic", "Magadi Road", "Srirampura",
    "Mantri Square Sampige Road", "Mahalakshmi", "Sandal Soap Factory",
    "Yeshwantpur", "Goraguntepalya", "Peenya", "Peenya Industry",
    "Jalahalli", "Dasarahalli", "Nagasandra", "Yelahanka",
    "Yelahanka Satellite Town", "Jakkur Cross", "Thanisandra",
    "Nagawara", "Kalyan Nagar", "SV Road", "Indiranagar",
    "Halasuru", "Trinity", "MG Road", "Cubbon Park",
    "Vidhana Soudha", "Sir M Visvesvaraya Station Central College",
    "Krantivira Sangolli Rayanna Railway Station",
    "National College", "Lalbagh", "South End Circle",
    "Jayanagar", "Rashtreeya Vidyalaya Road", "Banashankari",
    "Jayaprakash Nagar", "Yelachenahalli", "Konanakunte Cross",
    "Doddakallasandra", "Vajarahalli", "Anjanapura", "Gottigere",
    "Whitefield", "Kadugodi", "Channasandra", "Hoodi", "Garudacharpalya",
    "Domlur", "Indiranagar", "Swami Vivekananda Road", "Kalyan Nagar",
    "Nagawara", "Thanisandra", "Hebbal", "Kempegowda International Airport"
]

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html', stations=METRO_STATIONS)
    except Exception as e:
        print(f"Template error: {e}")
        # Return simple HTML if template fails
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bangalore Metro Booking</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
                .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                .button { background: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                .button:hover { background: #0056b3; }
                .auto-detect { color: #666; font-style: italic; margin: 10px 0; }
                .results { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
                .hidden { display: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚇 Bangalore Metro Booking</h1>
                <p>Voice-enabled ticket booking system with automatic language detection</p>
                <p class="auto-detect">🌍 Speaks in your language - English, Hindi, Marathi, Kannada, Tamil, Telugu</p>
                
                <button id="recordBtn" class="button" onclick="testRecording()">🎤 Click to Speak</button>
                
                <div id="status" style="margin-top: 20px;"></div>
                <div id="results" class="results hidden">
                    <h3>Results:</h3>
                    <div id="transcription"></div>
                    <div id="intent"></div>
                </div>
            </div>
            
            <script>
                let isRecording = false;
                let mediaRecorder = null;
                
                // Helper function to get language display name
                function getLanguageName(code) {
                    const names = {
                        'en': 'English',
                        'hi': 'Hindi',
                        'mr': 'Marathi', 
                        'kn': 'Kannada',
                        'ta': 'Tamil',
                        'te': 'Telugu'
                    };
                    return names[code] || code;
                }
                
                function testRecording() {
                    if (!isRecording) {
                        startRecording();
                    } else {
                        stopRecording();
                    }
                }
                
                async function startRecording() {
                    try {
                        document.getElementById('status').innerHTML = '🎤 Requesting microphone...';
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        
                        mediaRecorder = new MediaRecorder(stream);
                        let audioChunks = [];
                        
                        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                        mediaRecorder.onstop = () => processAudio(audioChunks);
                        
                        mediaRecorder.start();
                        isRecording = true;
                        
                        document.getElementById('recordBtn').innerHTML = '⏹️ Stop Recording';
                        document.getElementById('status').innerHTML = '🎤 Recording... Speak now!';
                        
                    } catch (error) {
                        document.getElementById('status').innerHTML = '❌ Microphone access denied: ' + error.message;
                    }
                }
                
                function stopRecording() {
                    if (mediaRecorder && isRecording) {
                        mediaRecorder.stop();
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        isRecording = false;
                        
                        document.getElementById('recordBtn').innerHTML = '🎤 Click to Speak';
                        document.getElementById('status').innerHTML = '🔄 Processing...';
                    }
                }
                
                async function processAudio(audioChunks) {
                    try {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'recording.wav');
                        // Language will be auto-detected
                        
                        const response = await fetch('/api/transcribe', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            // Create display for both romanized and native script
                            let transcriptionHtml = '<strong>You said:</strong> ' + result.transcription;
                            
                            // Add native script version if different from original
                            if (result.native_script && result.native_script !== result.transcription) {
                                transcriptionHtml += '<br><strong>In ' + getLanguageName(result.detected_language) + ':</strong> <span style="font-size: 1.2em; color: #0066cc;">' + result.native_script + '</span>';
                            }
                            
                            transcriptionHtml += '<br><small>🔍 Detected language: ' + getLanguageName(result.detected_language) + '</small>';
                            
                            document.getElementById('transcription').innerHTML = transcriptionHtml;
                            document.getElementById('results').classList.remove('hidden');
                            document.getElementById('status').innerHTML = '✅ Success!';
                            
                            // Process for intent
                            const intentResponse = await fetch('/api/process', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    text: result.transcription
                                    // language auto-detected
                                })
                            });
                            
                            const intentResult = await intentResponse.json();
                            
                            // Enhanced intent display
                            let intentHtml = '<strong>Intent:</strong> ' + intentResult.intent + ' (' + (intentResult.confidence * 100).toFixed(1) + '%)';
                            
                            // Add specific details based on intent type
                            if (intentResult.intent === 'book_ticket') {
                                if (intentResult.quantity && intentResult.quantity > 1) {
                                    intentHtml += '<br><strong>Quantity:</strong> ' + intentResult.quantity + ' tickets';
                                }
                                if (intentResult.from_station && intentResult.to_station) {
                                    intentHtml += '<br><strong>Route:</strong> ' + intentResult.from_station + ' → ' + intentResult.to_station;
                                }
                                if (intentResult.booking_id) {
                                    intentHtml += '<br><strong>Booking ID:</strong> ' + intentResult.booking_id;
                                }
                                if (intentResult.price) {
                                    intentHtml += '<br><strong>Total Price:</strong> ₹' + intentResult.price;
                                }
                                if (intentResult.status) {
                                    intentHtml += '<br><strong>Status:</strong> ' + intentResult.status;
                                }
                            } else if (intentResult.intent === 'fare_inquiry') {
                                if (intentResult.from_station && intentResult.to_station) {
                                    intentHtml += '<br><strong>Route:</strong> ' + intentResult.from_station + ' → ' + intentResult.to_station;
                                }
                                if (intentResult.quantity && intentResult.quantity > 1) {
                                    intentHtml += '<br><strong>For:</strong> ' + intentResult.quantity + ' passengers';
                                }
                                // Calculate estimated fare
                                if (intentResult.from_station && intentResult.to_station) {
                                    const baseFare = 10;
                                    const distanceFactor = Math.abs((intentResult.from_station.length % 20) - (intentResult.to_station.length % 20));
                                    const estimatedFare = (baseFare + distanceFactor) * (intentResult.quantity || 1);
                                    intentHtml += '<br><strong>Estimated Fare:</strong> ₹' + estimatedFare;
                                }
                            } else if (intentResult.intent === 'cancel_ticket') {
                                if (intentResult.booking_id) {
                                    intentHtml += '<br><strong>Booking ID:</strong> ' + intentResult.booking_id;
                                }
                                if (intentResult.stations && intentResult.stations.length > 0) {
                                    intentHtml += '<br><strong>Related Stations:</strong> ' + intentResult.stations.join(', ');
                                }
                                intentHtml += '<br><span style="color: orange;">⚠️ Cancellation request noted</span>';
                            } else if (intentResult.intent === 'route_inquiry') {
                                if (intentResult.from_station && intentResult.to_station) {
                                    intentHtml += '<br><strong>Route Query:</strong> ' + intentResult.from_station + ' → ' + intentResult.to_station;
                                    intentHtml += '<br><span style="color: blue;">ℹ️ Route information would be provided</span>';
                                }
                            } else if (intentResult.intent === 'booking_status') {
                                if (intentResult.booking_id) {
                                    intentHtml += '<br><strong>Booking ID:</strong> ' + intentResult.booking_id;
                                }
                                intentHtml += '<br><span style="color: blue;">ℹ️ Status check request noted</span>';
                            } else if (intentResult.intent === 'general_inquiry') {
                                intentHtml += '<br><span style="color: green;">ℹ️ General help request - assistance available</span>';
                            } else if (intentResult.intent === 'unknown') {
                                intentHtml += '<br><span style="color: red;">❓ Could not understand request. Please try again.</span>';
                            }
                            
                            document.getElementById('intent').innerHTML = intentHtml;
                            
                        } else {
                            document.getElementById('status').innerHTML = '❌ Error: ' + result.error;
                        }
                        
                    } catch (error) {
                        document.getElementById('status').innerHTML = '❌ Processing failed: ' + error.message;
                    }
                }
            </script>
        </body>
        </html>
        '''

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        # No language needed - will auto-detect from transcription
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save audio temporarily
        temp_dir = tempfile.gettempdir()
        audio_filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        audio_file.save(audio_path)
        
        print("🎤 Transcribing audio with auto-detection")
        
        # Transcribe using speech recognition (use English as base, then detect from text)
        transcription = speech_service.transcribe(audio_path, 'en')
        
        # Auto-detect language from transcription
        detected_language = language_service.detect_language(transcription)
        print(f"🔍 Detected language: {detected_language}")
        
        # Get native script version if applicable
        native_script_text = language_service.transliterate_to_native_script(transcription, detected_language)
        print(f"🎨 Native script: {native_script_text}")
        
        # Clean up
        os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'native_script': native_script_text,
            'detected_language': detected_language
        })
        
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_request():
    """Process text for intent detection and booking"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', None)  # None for auto-detection
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"🧠 Processing: '{text}' (language: {language or 'auto-detect'})")
        
        # Detect intent and extract information (with auto language detection)
        result = language_service.process_text(text, language)
        
        # If booking intent detected, calculate price and create booking
        if result['intent'] == 'book_ticket':
            from_station = result.get('from_station')
            to_station = result.get('to_station')
            quantity = result.get('quantity', 1)
            
            if from_station and to_station:
                # Simple pricing: base price + distance factor
                base_price = 10
                distance_factor = abs(hash(from_station) % 50 - hash(to_station) % 50)
                total_price = (base_price + distance_factor) * quantity
                
                booking_id = f"BM{uuid.uuid4().hex[:8].upper()}"
                
                result.update({
                    'booking_id': booking_id,
                    'price': total_price,
                    'status': 'confirmed'
                })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stations', methods=['GET'])
def get_stations():
    """Get all metro stations"""
    return jsonify({'stations': METRO_STATIONS})

if __name__ == '__main__':
    print("🚇 Starting Bangalore Metro Booking System...")
    print("📍 Available at: http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)
