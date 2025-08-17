#!/usr/bin/env python3
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Schéma minimal pour guider la sortie (optionnel)
INTENT_SCHEMA = {
    "name": "IntentResult",
    "schema": {
        "type": "object",
        "required": ["intent_type", "raw_user_message"],
        "properties": {
            "intent_type": {"type": "string"},
            "raw_user_message": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "additionalProperties": False,
    },
    "strict": True,
}

QUESTIONS = [
    "Quel est le solde de mon compte courant ?",
    "Combien ai-je dépensé chez Amazon en 2024 ?",
    "Bonjour",
]

def main(model="gpt-4.1-mini"):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for q in QUESTIONS:
        start = time.perf_counter()
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Tu es un classifieur d’intentions financières."},
                {"role": "user", "content": q},
            ],
            text={"format": "json_schema", "json_schema": INTENT_SCHEMA},
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        print(f"\nQ: {q}")
        print(f"Latence: {latency_ms} ms")
        print("Résultat structuré:", resp.output_parsed)

if __name__ == "__main__":
    main()
