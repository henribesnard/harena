#!/usr/bin/env python3
"""
Script de test pour vérifier que le mapping Netflix fonctionne avec les templates corrigés
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

def test_netflix_query():
    """Test de la requête Netflix"""
    print("Test du mapping Netflix avec templates corriges...")
    
    # Créer un token JWT
    token = create_jwt_token(1)
    print(f"Token JWT cree: {token[:50]}...")
    
    # Préparer la requête
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "message": "Mes dépenses Netflix ?"
    }
    
    print(f"Envoi requete: {data}")
    
    try:
        # Faire la requête
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Reponse recue:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Vérifier si Netflix apparaît dans la réponse
            response_text = json.dumps(result, ensure_ascii=False).lower()
            if "netflix" in response_text:
                print("SUCCESS: Netflix trouve dans la reponse!")
            else:
                print("FAILURE: Netflix non trouve dans la reponse")
                
        else:
            print(f"Erreur HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Erreur lors de la requete: {str(e)}")

if __name__ == "__main__":
    test_netflix_query()