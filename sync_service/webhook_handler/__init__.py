"""
Package gestionnaire de webhooks Bridge.

Ce package est responsable de la réception, validation et traitement 
des événements webhook envoyés par Bridge API.
"""

# Exporter les fonctions principales pour faciliter l'import
from sync_service.webhook_handler.processor import process_webhook, validate_webhook