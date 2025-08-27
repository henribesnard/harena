"""Exemples few-shot pour l'extraction d'entités financières."""

ENTITY_EXTRACTION_FEW_SHOT_EXAMPLES = [
    {
        "input": "J'ai dépensé 20€ chez Carrefour hier.",
        "output": {
            "entities": {
                "amounts": [{"value": 20, "currency": "EUR", "operator": "eq"}],
                "dates": [{"type": "relative", "value": "hier", "text": "hier"}],
                "merchants": ["Carrefour"],
                "categories": [],
                "operation_types": [],
                "text_search": []
            }
        }
    },
    {
        "input": "Montre-moi mes paiements supérieurs à 100 euros en juillet.",
        "output": {
            "entities": {
                "amounts": [{"value": 100, "currency": "EUR", "operator": "gt"}],
                "dates": [{"type": "month", "value": "2024-07", "text": "juillet"}],
                "merchants": [],
                "categories": [],
                "operation_types": ["paiement"],
                "text_search": []
            }
        }
    }
]
