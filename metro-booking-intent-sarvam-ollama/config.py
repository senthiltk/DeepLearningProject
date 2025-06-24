import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#API Keys & Endpoints
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") # Uing this for RAG

SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
SARVAM_TRANSLATE_ENDPOINT = "https://api.sarvam.ai/translate"

#Audio Recording Settings
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SECONDS = 5

#Supported Languages for Sarvam AI
SARVAM_STT_LANGUAGES = [
    "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "or-IN", "pa-IN", "ta-IN", "te-IN"
]
SARVAM_TRANSLATE_LANGUAGES = [
    "auto", "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "or-IN", "pa-IN", "ta-IN", "te-IN"
]

#RAG Knowledge Base
RAG_KNOWLEDGE_BASE = [
    {
        "intent_name": "Book Metro Ticket",
        "description": "The user wants to purchase a metro ticket, request a metro ride, or inquire about booking a journey.",
        "examples": [
            "I want to book a metro ticket.",
            "Book me a ticket to the airport.",
            "Can I get a metro ticket from Varthur to Majestic?",
            "I need to book a ride for tomorrow.",
            "Metro ticket booking.",
            "Please book a journey for me.",
            "How do I buy a metro ticket?",
            "Book a train ticket to Indiranagar."
        ]
    },
    {
        "intent_name": "Check Metro Balance",
        "description": "The user wants to check the balance of their metro card or inquire about remaining credit.",
        "examples": [
            "What's my metro card balance?",
            "Check my balance.",
            "How much money is on my card?",
            "Inquire about my metro credit.",
            "Balance check."
        ]
    },
    {
        "intent_name": "General Query",
        "description": "The user is asking a general question not related to metro booking, balance, or cancellations.",
        "examples": [
            "Hello.",
            "How are you?",
            "What's the weather like?",
            "Tell me a joke.",
            "Who are you?",
            "Can you help me with something else?"
        ]
    },
    {
        "intent_name": "Cancel Booking",
        "description": "The user wants to cancel a previously made metro ticket booking.",
        "examples": [
            "Cancel my metro ticket.",
            "I want to cancel my booking.",
            "Undo the last metro reservation."
        ]
    }
]