#!/usr/bin/env python3
"""
Test simple et rapide de la clé API OpenAI
"""

import os
from pathlib import Path

print("🧪 TEST RAPIDE API OPENAI")
print("="*40)

# 1. Charger le fichier .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Fichier .env chargé avec python-dotenv")
except ImportError:
    print("⚠️ python-dotenv non installé, chargement manuel...")
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        print("✅ Variables .env chargées manuellement")
    else:
        print("❌ Fichier .env introuvable")
        exit(1)

# 2. Vérifier la clé
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ OPENAI_API_KEY non trouvée")
    print("Vérifiez votre fichier .env")
    exit(1)

print(f"✅ Clé API trouvée: {api_key[:8]}...")

# 3. Tester OpenAI
try:
    from openai import OpenAI
    print("✅ Module openai importé")
    
    client = OpenAI()  # Utilise automatiquement OPENAI_API_KEY
    print("✅ Client OpenAI créé")
    
    # Test simple
    print("🔄 Test d'appel API...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    
    print("✅ Test API réussi!")
    print(f"   Réponse: {response.choices[0].message.content}")
    print(f"   Tokens: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    
    # Diagnostics spécifiques
    error_str = str(e).lower()
    if "invalid api key" in error_str:
        print("💡 Clé API invalide")
    elif "quota" in error_str or "billing" in error_str:
        print("💡 Problème de facturation/quota")
    elif "rate limit" in error_str:
        print("💡 Limite de taux atteinte")
    
    exit(1)

print("\n🎉 Votre clé OpenAI fonctionne parfaitement!")
print("\n💡 Pour corriger votre script evaluate_templates.py,")
print("   ajoutez ces lignes au début du fichier:")
print()
print("   # Charger .env AVANT d'importer OpenAI")
print("   try:")
print("       from dotenv import load_dotenv")
print("       load_dotenv()")
print("   except ImportError:")
print("       pass")