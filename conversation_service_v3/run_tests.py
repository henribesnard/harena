#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test exhaustif pour conversation_service_v3
Execute 22 questions et génère un rapport détaillé
"""
import sys
import io
import requests
import json
import time
from datetime import datetime

# Force UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
API_URL = "http://localhost:3008/api/v3/conversation/3"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjE2NTQzOTYsInN1YiI6IjMiLCJwZXJtaXNzaW9ucyI6WyJjaGF0OndyaXRlIl19.oIXndbsYS8_CVUjII6znskomGRq9DxHUaiX8cmpjpOA"

# Charger les questions
with open('test_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Résultats
results = []

print(f"Starting test suite with {len(questions)} questions...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for i, test in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] Testing: {test['question']}")

    # Préparer la requête
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": test['question']
    }

    # Mesurer le temps
    start_time = time.time()

    try:
        # Exécuter la requête
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        # Calculer la latence
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            # Extraire les informations
            result = {
                "id": test['id'],
                "category": test['category'],
                "question": test['question'],
                "status": "SUCCESS",
                "latency_seconds": round(latency, 2),
                "total_results": data.get('metadata', {}).get('total_results', 0),
                "pipeline_time_ms": data.get('metadata', {}).get('pipeline_time_ms', 0),
                "aggregations_requested": data.get('metadata', {}).get('query_analysis', {}).get('aggregations_requested', []),
                "response_preview": data.get('response', {}).get('message', '')[:300] + "...",
                "full_response": data.get('response', {}).get('message', ''),
                "query_generated": data.get('metadata', {}).get('query_generated', {}),
            }

            print(f"  [OK] Success - {latency:.2f}s - {result['total_results']} results")

        else:
            result = {
                "id": test['id'],
                "category": test['category'],
                "question": test['question'],
                "status": f"FAILED ({response.status_code})",
                "latency_seconds": round(latency, 2),
                "error": response.text
            }
            print(f"  [FAIL] Failed - {response.status_code}")

    except Exception as e:
        latency = time.time() - start_time
        result = {
            "id": test['id'],
            "category": test['category'],
            "question": test['question'],
            "status": "ERROR",
            "latency_seconds": round(latency, 2),
            "error": str(e)
        }
        print(f"  [ERROR] Error - {str(e)}")

    results.append(result)

    # Petite pause entre les requêtes
    time.sleep(1)

# Sauvegarder les résultats
output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[DONE] Tests completed! Results saved to {output_file}")

# Statistiques
success_count = sum(1 for r in results if "SUCCESS" in r['status'])
total_count = len(results)
avg_latency = sum(r['latency_seconds'] for r in results if 'latency_seconds' in r) / total_count

print(f"\nStatistics:")
print(f"  Total tests: {total_count}")
print(f"  Successful: {success_count} ({success_count/total_count*100:.1f}%)")
print(f"  Failed: {total_count - success_count}")
print(f"  Average latency: {avg_latency:.2f}s")
