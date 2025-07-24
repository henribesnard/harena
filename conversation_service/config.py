"""Configuration minimaliste"""
import os
from typing import Dict, Any

# Modèle TinyBERT
TINYBERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
DEVICE = "cpu"  # ou "cuda" si GPU disponible
MAX_LENGTH = 128

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8001
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Intentions financières supportées
FINANCIAL_INTENTS = {
    "BALANCE_CHECK": "balance",
    "TRANSFER": "transfer", 
    "EXPENSE_ANALYSIS": "expense",
    "CARD_MANAGEMENT": "card",
    "GREETING": "greeting",
    "HELP": "help",
    "GOODBYE": "goodbye",
    "UNKNOWN": "unknown"
}

# Seuils de confiance
CONFIDENCE_THRESHOLD = 0.5