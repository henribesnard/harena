#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script - runs 8 representative questions and generates immediate report
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

# 8 representative questions covering all categories
test_questions = [
    {"id": 1, "category": "Agregations simples", "question": "Combien j'ai depense en total"},
    {"id": 2, "category": "Agregations simples", "question": "Mes depenses de plus de 50 euros"},
    {"id": 3, "category": "Agregations par categorie", "question": "Repartition de mes depenses par categorie"},
    {"id": 4, "category": "Agregations par marchand", "question": "Ou je depense le plus"},
    {"id": 5, "category": "Agregations temporelles", "question": "Evolution mensuelle de mes depenses"},
    {"id": 6, "category": "Filtres complexes", "question": "Mes depenses alimentaires de plus de 20 euros"},
    {"id": 7, "category": "Recherches specifiques", "question": "Mes virements sortants"},
    {"id": 8, "category": "Statistiques globales", "question": "Resume de mes finances"},
]

results = []

print(f"Starting quick test suite with {len(test_questions)} representative questions...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
sys.stdout.flush()

for i, test in enumerate(test_questions, 1):
    print(f"[{i}/{len(test_questions)}] Testing: {test['question']}")
    sys.stdout.flush()

    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": test['question']
    }

    start_time = time.time()

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            result = {
                "id": test['id'],
                "category": test['category'],
                "question": test['question'],
                "status": "SUCCESS",
                "latency_seconds": round(latency, 2),
                "total_results": data.get('metadata', {}).get('total_results', 0),
                "pipeline_time_ms": data.get('metadata', {}).get('pipeline_time_ms', 0),
                "aggregations_requested": data.get('metadata', {}).get('query_analysis', {}).get('aggregations_requested', []),
                "aggregations_summary": data.get('response', {}).get('structured_data', {}).get('aggregations_summary', 'N/A'),
                "response_preview": data.get('response', {}).get('message', '')[:200] + "...",
                "full_response": data.get('response', {}).get('message', ''),
                "query_generated": data.get('metadata', {}).get('query_generated', {}),
            }

            print(f"  [OK] Success - {latency:.2f}s - {result['total_results']} results")
            sys.stdout.flush()

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
            sys.stdout.flush()

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
        sys.stdout.flush()

    results.append(result)
    time.sleep(1)  # Small pause between requests

# Save results
output_file = f"test_results_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[DONE] Tests completed! Results saved to {output_file}")
sys.stdout.flush()

# Statistics
success_count = sum(1 for r in results if "SUCCESS" in r['status'])
total_count = len(results)
avg_latency = sum(r['latency_seconds'] for r in results if 'latency_seconds' in r) / total_count

print(f"\nStatistics:")
print(f"  Total tests: {total_count}")
print(f"  Successful: {success_count} ({success_count/total_count*100:.1f}%)")
print(f"  Failed: {total_count - success_count}")
print(f"  Average latency: {avg_latency:.2f}s")
sys.stdout.flush()
