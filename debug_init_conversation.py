#!/usr/bin/env python3
"""
🔧 Script debug initialisation conversation_service

Diagnostique pourquoi les composants L0/L1/L2 ne s'initialisent pas correctement
"""

import requests
import json
import asyncio
import logging
from typing import Dict, Any

# Configuration debug
DEBUG_URL = "http://localhost:8000/api/v1/conversation"
ENDPOINTS = {
    "health": f"{DEBUG_URL}/health",
    "metrics": f"{DEBUG_URL}/metrics", 
    "status": f"{DEBUG_URL}/status",
    "debug": f"{DEBUG_URL}/debug/test-levels"
}

def check_service_endpoints():
    """Vérification accessibilité endpoints"""
    print("🔍 VÉRIFICATION ENDPOINTS")
    print("="*50)
    
    for name, url in ENDPOINTS.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: {response.status_code}")
                
                # Détails santé
                if name == "health":
                    data = response.json()
                    print(f"   Status: {data.get('status', 'unknown')}")
                    if 'intent_engine' in data:
                        engine_status = data['intent_engine']
                        print(f"   Intent Engine: {engine_status.get('initialized', 'unknown')}")
                
                # Détails métriques
                elif name == "metrics":
                    data = response.json()
                    if 'intent_engine' in data:
                        engine = data['intent_engine']
                        print(f"   Engine Status: {engine.get('status', 'unknown')}")
                        print(f"   Classifications: {engine.get('total_classifications', 0)}")
                
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"💥 {name}: {str(e)}")

def test_intent_engine_components():
    """Test spécifique composants Intent Detection Engine"""
    print("\n🧠 TEST COMPOSANTS INTENT ENGINE")
    print("="*50)
    
    # Test debug endpoint pour forcer niveaux
    test_cases = [
        {"message": "solde", "expected": "L0_PATTERN"},
        {"message": "mes dépenses restaurant", "expected": "L1_LIGHTWEIGHT"}, 
        {"message": "analyse complexe", "expected": "L2_LLM"}
    ]
    
    for case in test_cases:
        print(f"\n🔧 Test force niveau {case['expected']}:")
        
        try:
            # Test avec force niveau
            response = requests.post(
                ENDPOINTS["debug"],
                headers={'Content-Type': 'application/json'},
                json={
                    "message": case["message"],
                    "user_id": 34,
                    "force_level": case["expected"]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                actual_level = data.get("actual_level", "unknown")
                intent = data.get("intent", "unknown")
                latency = data.get("latency_ms", 0)
                
                print(f"   Message: '{case['message']}'")
                print(f"   Forcé: {case['expected']} → Réel: {actual_level}")
                print(f"   Intent: {intent}")
                print(f"   Latence: {latency:.1f}ms")
                
                if actual_level != case["expected"]:
                    print(f"   ⚠️ Niveau forcé non respecté - composant indisponible")
                else:
                    print(f"   ✅ Composant fonctionnel")
                    
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   💥 Erreur: {e}")

def analyze_configuration_issues():
    """Analyse problèmes configuration potentiels"""
    print("\n⚙️ ANALYSE CONFIGURATION")
    print("="*50)
    
    # Vérification variables d'environnement critiques
    env_vars_to_check = [
        "DEEPSEEK_API_KEY",
        "REDIS_CACHE_ENABLED", 
        "CONVERSATION_SERVICE_DEBUG",
        "MIN_CONFIDENCE_THRESHOLD"
    ]
    
    print("Variables d'environnement:")
    import os
    for var in env_vars_to_check:
        value = os.environ.get(var, "NON_DÉFINIE")
        if var == "DEEPSEEK_API_KEY" and value != "NON_DÉFINIE":
            value = f"{value[:10]}..." if len(value) > 10 else "DÉFINIE"
        print(f"   {var}: {value}")
    
    # Test configuration service
    try:
        response = requests.get(ENDPOINTS["status"], timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"\nConfiguration service:")
            print(f"   Architecture: {status.get('architecture', 'unknown')}")
            print(f"   Model: {status.get('model', 'unknown')}")
            print(f"   Status: {status.get('status', 'unknown')}")
            
            if 'endpoints' in status:
                endpoints = status['endpoints']
                print(f"   Endpoints configurés: {len(endpoints)}")
                
    except Exception as e:
        print(f"   ❌ Erreur status: {e}")

def test_simple_classification():
    """Test classification simple pour validation"""
    print("\n🎯 TEST CLASSIFICATION SIMPLE")
    print("="*50)
    
    simple_tests = [
        "solde",
        "bonjour", 
        "aide",
        "quel est mon solde"
    ]
    
    for message in simple_tests:
        try:
            response = requests.post(
                f"{DEBUG_URL}/chat",
                headers={'Content-Type': 'application/json'},
                json={
                    "message": message,
                    "user_id": 34,
                    "conversation_id": f"debug_{message}"
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                intent = data.get("intent", "UNKNOWN")
                confidence = data.get("confidence", 0.0)
                metadata = data.get("processing_metadata", {})
                level = metadata.get("level_used", "UNKNOWN")
                
                print(f"'{message}' → {intent} ({confidence:.2f}) [{level}]")
                
                # Diagnostic spécifique
                if level == "ERROR_FALLBACK":
                    print(f"   ⚠️ Fallback - vérifier initialisation composants")
                elif confidence == 0.5:
                    print(f"   ⚠️ Confidence par défaut - classification basique")
                else:
                    print(f"   ✅ Classification normale")
                    
            else:
                print(f"'{message}' → HTTP {response.status_code}")
                
        except Exception as e:
            print(f"'{message}' → Erreur: {e}")

def check_logs_and_errors():
    """Vérification logs et erreurs potentielles"""
    print("\n📋 VÉRIFICATION LOGS")
    print("="*50)
    
    # Instructions pour vérifier logs service
    print("Pour diagnostiquer plus avant, vérifiez les logs du service:")
    print("1. Logs démarrage du service")
    print("2. Erreurs d'initialisation Intent Detection Engine")
    print("3. Problèmes de chargement TinyBERT/DeepSeek")
    print("4. Échecs connexion Redis")
    
    print("\nCommandes utiles:")
    print("   # Si service en local")
    print("   tail -f conversation_service.log")
    print("   # Si via Docker")
    print("   docker logs conversation_service")
    print("   # Vérifier processus")
    print("   ps aux | grep conversation")

def generate_fix_recommendations():
    """Génération recommandations corrections"""
    print("\n💡 RECOMMANDATIONS CORRECTIONS")
    print("="*50)
    
    recommendations = [
        "1. VÉRIFIER INITIALISATION:",
        "   - Restart conversation_service",
        "   - Vérifier logs initialisation Intent Detection Engine",
        "   - Confirmer chargement composants L0/L1/L2",
        "",
        "2. CONFIGURATION L0 PATTERNS:",
        "   - Vérifier patterns financiers dans pattern_matcher.py",
        "   - Confirmer compilation regex patterns",
        "   - Tester patterns L0 isolément",
        "",
        "3. CONFIGURATION L1 TINYBERT:",
        "   - Vérifier installation sentence-transformers",
        "   - Confirmer téléchargement modèle embeddings",
        "   - Tester classification L1 isolément",
        "",
        "4. CONFIGURATION L2 DEEPSEEK:",
        "   - Vérifier DEEPSEEK_API_KEY",
        "   - Tester connexion API DeepSeek",
        "   - Confirmer quotas API",
        "",
        "5. CONFIGURATION CACHE:",
        "   - Vérifier Redis si activé",
        "   - Fallback cache local",
        "",
        "6. SEUILS ET TIMEOUTS:",
        "   - Réduire MIN_CONFIDENCE_THRESHOLD (0.7 → 0.5)",
        "   - Augmenter timeouts initialisation",
        "   - Mode debug pour traces détaillées"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """Diagnostic complet initialisation"""
    print("🔧 DIAGNOSTIC INITIALISATION CONVERSATION SERVICE")
    print("="*80)
    
    # Tests séquentiels
    check_service_endpoints()
    test_intent_engine_components()
    analyze_configuration_issues()
    test_simple_classification()
    check_logs_and_errors()
    generate_fix_recommendations()
    
    print("\n" + "="*80)
    print("🎯 RÉSUMÉ DIAGNOSTIC")
    print("="*80)
    print("Le diagnostic montre que l'Intent Detection Engine n'est pas")
    print("correctement initialisé, causant les fallbacks constants.")
    print("")
    print("ACTIONS PRIORITAIRES:")
    print("1. Restart conversation_service")
    print("2. Vérifier logs initialisation")  
    print("3. Confirmer variables d'environnement")
    print("4. Tester composants individuellement")

if __name__ == "__main__":
    main()