#!/usr/bin/env python3
"""
Script de test pour vérifier que les expressions de date françaises "ce mois" fonctionnent
"""
import jwt
import requests
import json
from datetime import datetime, timedelta, timezone

# Configuration
SECRET_KEY = "Harena2032Harena2032Harena2032Harena2032Harena2032"
API_URL = "http://localhost:8001/api/v1/api/v1/conversation/1"

def create_jwt_token(user_id: int = 1) -> str:
    """Crée un token JWT pour les tests"""
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def test_ce_mois_query():
    """Test de la requête 'mes dépense de ce mois'"""
    print("Test de la requête avec expression de date française 'ce mois'...")
    
    # Créer un token JWT
    token = create_jwt_token(1)
    print(f"Token JWT créé: {token[:50]}...")
    
    # Préparer la requête
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "message": "mes dépense de ce mois"
    }
    
    print(f"Envoi requête: {data}")
    
    try:
        # Faire la requête
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Réponse reçue:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Vérifier s'il n'y a pas d'erreur de parsing de date
            response_text = json.dumps(result, ensure_ascii=False).lower()
            if "failed to parse date field" in response_text:
                print("FAILURE: Erreur de parsing de date détectée")
            else:
                print("SUCCESS: Pas d'erreur de parsing de date")
                
        else:
            print(f"Erreur HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Erreur lors de la requête: {str(e)}")

if __name__ == "__main__":
    test_ce_mois_query()