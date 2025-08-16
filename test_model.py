#!/usr/bin/env python3
"""
Script de test corrigé pour Phi-3.5 - Détection d'intention
Version optimisée avec meilleur prompt et gestion d'erreurs
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import argparse
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("\n📦 Installation des dépendances requises...")
    print("Exécutez : pip install transformers torch pydantic accelerate sentencepiece protobuf")
    sys.exit(1)

# ==================== CONFIGURATION ====================

# Dataset complet du mock Harena
MOCK_INTENT_RESPONSES = {
    "Mes transactions Netflix ce mois": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.95,
        "entities": [
            {"entity_type": "MERCHANT", "raw_value": "Netflix", "normalized_value": "netflix", "confidence": 0.98},
            {"entity_type": "RELATIVE_DATE", "raw_value": "ce mois", "normalized_value": "current_month", "confidence": 0.90}
        ]
    },
    "Combien j'ai dépensé chez Carrefour ?": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.93,
        "entities": [
            {"entity_type": "MERCHANT", "raw_value": "Carrefour", "normalized_value": "carrefour", "confidence": 0.97}
        ]
    },
    "Mes achats de plus de 50 euros": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.92,
        "entities": [
            {"entity_type": "AMOUNT", "raw_value": "50 euros", "normalized_value": 50.0, "confidence": 0.95}
        ]
    },
    "Quelles sont mes dépenses restaurant ?": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.91,
        "entities": [
            {"entity_type": "CATEGORY", "raw_value": "restaurant", "normalized_value": "restaurant", "confidence": 0.94}
        ]
    },
    "Quel est mon solde ?": {
        "intent_type": "ACCOUNT_BALANCE",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.96,
        "entities": []
    },
    "Bonjour": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "GREETING",
        "confidence": 0.97,
        "entities": []
    }
}

# ==================== MODÈLES PYDANTIC SIMPLIFIÉS ====================

class IntentCategory(str, Enum):
    FINANCIAL_QUERY = "FINANCIAL_QUERY"
    GREETING = "GREETING"
    FILTER_REQUEST = "FILTER_REQUEST"
    GOAL_TRACKING = "GOAL_TRACKING"
    CONFIRMATION = "CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    UNKNOWN = "UNKNOWN"

class EntityType(str, Enum):
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    MERCHANT = "MERCHANT"
    CATEGORY = "CATEGORY"
    ACCOUNT = "ACCOUNT"
    RELATIVE_DATE = "RELATIVE_DATE"
    TRANSACTION_TYPE = "TRANSACTION_TYPE"

class FinancialEntity(BaseModel):
    entity_type: EntityType
    raw_value: str
    normalized_value: Any
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)

class IntentResult(BaseModel):
    intent_type: str
    intent_category: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    entities: List[FinancialEntity] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    raw_user_message: Optional[str] = None

# ==================== DÉTECTEUR CORRIGÉ ====================

class ImprovedIntentDetector:
    """Détecteur amélioré avec meilleur prompt et gestion d'erreurs"""

    def __init__(
        self,
        use_model: bool = True,
        debug: bool = False,
        model_name: Optional[str] = None,
    ):
        """Initialise le détecteur.

        Args:
            use_model: Active le chargement du modèle si ``True``.
            debug: Active les sorties de debug.
            model_name: Nom ou chemin du modèle à charger. Peut également être
                fourni via la variable d'environnement ``MODEL_NAME``.
        """

        self.use_model = use_model
        self.debug = debug
        self.model = None
        self.tokenizer = None
        # Permet de définir le modèle via paramètre, variable d'env ou valeur par défaut
        self.model_name = model_name or os.getenv(
            "MODEL_NAME", "microsoft/Phi-3.5-mini-instruct"
        )

        if use_model:
            print(f"🚀 Chargement du modèle {self.model_name}...")
            try:
                self._load_model(self.model_name)
                print("✅ Modèle chargé avec succès\n")
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle : {e}")
                print("📌 Basculement en mode mock uniquement\n")
                self.use_model = False

    def _load_model(self, model_name: str):
        """Charge le modèle avec configuration optimisée"""

        # Tokenizer avec configuration correcte
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'  # Important pour Phi-3.5
        )

        # Fix du pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Évite le warning flash-attention
        )
        self.model.eval()
    
    def _create_enhanced_prompt(self, query: str) -> str:
        """Prompt amélioré avec few-shot examples"""
        
        # Prompt structuré pour Phi-3.5
        prompt = """<|system|>
You are a financial intent classifier. Analyze the user query and return a JSON object.

Intent types: TRANSACTION_SEARCH, SPENDING_ANALYSIS, ACCOUNT_BALANCE, BUDGET_TRACKING, CONVERSATIONAL
Categories: FINANCIAL_QUERY, GREETING, FILTER_REQUEST, GOAL_TRACKING
Entity types: AMOUNT, DATE, MERCHANT, CATEGORY, ACCOUNT

Example 1:
Query: "Show my Netflix transactions"
Output: {"intent_type": "TRANSACTION_SEARCH", "intent_category": "FINANCIAL_QUERY", "confidence": 0.95, "entities": [{"entity_type": "MERCHANT", "raw_value": "Netflix", "normalized_value": "netflix", "confidence": 0.98}]}

Example 2:
Query: "What's my balance?"
Output: {"intent_type": "ACCOUNT_BALANCE", "intent_category": "FINANCIAL_QUERY", "confidence": 0.96, "entities": []}

Example 3:
Query: "Hello"
Output: {"intent_type": "CONVERSATIONAL", "intent_category": "GREETING", "confidence": 0.97, "entities": []}

IMPORTANT: Return ONLY the JSON object, no explanations.<|end|>
<|user|>
Query: "{}"
<|end|>
<|assistant|>
""".format(query)
        
        return prompt
    
    def _extract_json_safely(self, response: str) -> Dict[str, Any]:
        """Extraction JSON robuste avec multiples stratégies"""
        
        if self.debug:
            print(f"DEBUG - Raw response: {response[:200]}...")
        
        # Nettoyer la réponse
        response = response.strip()
        
        # Stratégie 1: JSON direct
        try:
            return json.loads(response)
        except:
            pass
        
        # Stratégie 2: Trouver le JSON entre accolades
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Stratégie 3: Essayer de réparer le JSON
        try:
            # Remplacer les single quotes par double quotes
            fixed = response.replace("'", '"')
            return json.loads(fixed)
        except:
            pass
        
        # Stratégie 4: Parser manuel pour cas simples
        if "TRANSACTION_SEARCH" in response:
            return {
                "intent_type": "TRANSACTION_SEARCH",
                "intent_category": "FINANCIAL_QUERY",
                "confidence": 0.7,
                "entities": []
            }
        elif "SPENDING_ANALYSIS" in response:
            return {
                "intent_type": "SPENDING_ANALYSIS",
                "intent_category": "FINANCIAL_QUERY",
                "confidence": 0.7,
                "entities": []
            }
        elif "ACCOUNT_BALANCE" in response:
            return {
                "intent_type": "ACCOUNT_BALANCE",
                "intent_category": "FINANCIAL_QUERY",
                "confidence": 0.7,
                "entities": []
            }
        elif "CONVERSATIONAL" in response or "GREETING" in response:
            return {
                "intent_type": "CONVERSATIONAL",
                "intent_category": "GREETING",
                "confidence": 0.7,
                "entities": []
            }
        
        # Échec total
        raise ValueError(f"Impossible d'extraire JSON de: {response[:100]}")
    
    @torch.no_grad()
    def detect_with_model(self, query: str) -> IntentResult:
        """Détection améliorée avec gestion d'erreurs"""
        start = time.time()
        
        try:
            # Prompt amélioré
            prompt = self._create_enhanced_prompt(query)
            
            # Tokenisation avec attention au padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # Génération avec paramètres optimisés
            generation_config = {
                "max_new_tokens": 150,  # Plus de tokens
                "temperature": 0.1,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "num_beams": 1  # Greedy pour vitesse
            }
            
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
            
            # Décoder uniquement la partie générée
            generated_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            if self.debug:
                print(f"DEBUG - Generated: {response}")
            
            # Extraire et parser le JSON
            result_dict = self._extract_json_safely(response)
            
            # Créer les entités
            entities = []
            for ent in result_dict.get("entities", []):
                try:
                    entities.append(FinancialEntity(
                        entity_type=EntityType(ent["entity_type"]),
                        raw_value=ent.get("raw_value", ""),
                        normalized_value=ent.get("normalized_value", ent.get("raw_value", "")),
                        confidence=ent.get("confidence", 0.9)
                    ))
                except:
                    continue  # Ignorer les entités mal formées
            
            # Valider la catégorie
            try:
                category = IntentCategory(result_dict.get("intent_category", "UNKNOWN"))
            except:
                category = IntentCategory.UNKNOWN
            
            return IntentResult(
                intent_type=result_dict.get("intent_type", "UNKNOWN"),
                intent_category=category,
                confidence=result_dict.get("confidence", 0.5),
                entities=entities,
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query
            )
            
        except Exception as e:
            if self.debug:
                print(f"DEBUG - Error: {e}")
            
            # Fallback intelligent basé sur mots-clés
            return self._fallback_detection(query, (time.time() - start) * 1000)
    
    def _fallback_detection(self, query: str, processing_time: float) -> IntentResult:
        """Détection de fallback basée sur mots-clés"""
        query_lower = query.lower()
        
        # Règles simples
        if any(word in query_lower for word in ["solde", "balance", "combien j'ai"]):
            return IntentResult(
                intent_type="ACCOUNT_BALANCE",
                intent_category=IntentCategory.FINANCIAL_QUERY,
                confidence=0.6,
                entities=[],
                processing_time_ms=processing_time,
                raw_user_message=query
            )
        elif any(word in query_lower for word in ["dépensé", "dépenses", "achats"]):
            return IntentResult(
                intent_type="SPENDING_ANALYSIS",
                intent_category=IntentCategory.FINANCIAL_QUERY,
                confidence=0.6,
                entities=[],
                processing_time_ms=processing_time,
                raw_user_message=query
            )
        elif any(word in query_lower for word in ["transaction", "paiement", "achat"]):
            return IntentResult(
                intent_type="TRANSACTION_SEARCH",
                intent_category=IntentCategory.FINANCIAL_QUERY,
                confidence=0.6,
                entities=[],
                processing_time_ms=processing_time,
                raw_user_message=query
            )
        elif any(word in query_lower for word in ["bonjour", "salut", "hello"]):
            return IntentResult(
                intent_type="CONVERSATIONAL",
                intent_category=IntentCategory.GREETING,
                confidence=0.8,
                entities=[],
                processing_time_ms=processing_time,
                raw_user_message=query
            )
        else:
            return IntentResult(
                intent_type="UNKNOWN",
                intent_category=IntentCategory.UNKNOWN,
                confidence=0.3,
                entities=[],
                processing_time_ms=processing_time,
                raw_user_message=query
            )
    
    def detect_with_mock(self, query: str) -> IntentResult:
        """Détection avec données mock"""
        start = time.time()
        
        if query in MOCK_INTENT_RESPONSES:
            data = MOCK_INTENT_RESPONSES[query]
            
            entities = []
            for ent in data.get("entities", []):
                entities.append(FinancialEntity(
                    entity_type=EntityType(ent["entity_type"]),
                    raw_value=ent["raw_value"],
                    normalized_value=ent["normalized_value"],
                    confidence=ent["confidence"]
                ))
            
            return IntentResult(
                intent_type=data["intent_type"],
                intent_category=IntentCategory(data["intent_category"]),
                confidence=data["confidence"],
                entities=entities,
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query
            )
        else:
            return IntentResult(
                intent_type="UNKNOWN",
                intent_category=IntentCategory.UNKNOWN,
                confidence=0.0,
                entities=[],
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query
            )
    
    def detect(self, query: str) -> Tuple[IntentResult, IntentResult]:
        """Détecte avec mock et modèle"""
        mock_result = self.detect_with_mock(query)
        
        if self.use_model:
            model_result = self.detect_with_model(query)
        else:
            model_result = None

        return mock_result, model_result

# ==================== ÉVALUATION ====================

def evaluate(detector, dataset):
    """Exécute la boucle de test et retourne les prédictions et succès."""

    test_questions = list(dataset.keys())

def main(use_model: bool = False, debug: bool = False):

def main(model_name: str):
    """Test amélioré avec meilleure gestion d'erreurs"""

    print("=" * 80)
    print("🧪 TEST AMÉLIORÉ - DÉTECTION D'INTENTION PHI-3.5")
    print("=" * 80)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    mode_desc = "Mock + Modèle" if use_model else "Mock uniquement"
    if debug:
        mode_desc += " (DEBUG)"
    print(f"Mode : {mode_desc}\n")

    detector = ImprovedIntentDetector(use_model=use_model, debug=debug)
    
    print("🔧 OPTIONS DE TEST:")
    print("1. Mock uniquement (rapide)")
    print("2. Mock + Modèle (nécessite 4-6GB RAM)")
    print("3. Mock + Modèle avec DEBUG")
    print()
    
    choice = input("Votre choix (1, 2 ou 3) : ").strip()
    use_model = choice in ["2", "3"]
    debug = choice == "3"
    
    print()
    detector = ImprovedIntentDetector(
        use_model=use_model, debug=debug, model_name=model_name
    )
    
    # Questions de test sélectionnées
    test_questions = list(MOCK_INTENT_RESPONSES.keys())
    
    print(f"\n🚀 TEST DE {len(test_questions)} QUESTIONS\n")

    successes = 0
    latencies = []
    predictions = {}

    for i, query in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] 💬 {query}")
        print("-" * 60)

        mock_result, model_result = detector.detect(query)

        expected_intent = mock_result.intent_type
        print(f"📌 Attendu : {expected_intent} ({mock_result.confidence:.2f})")

        if model_result:
            predicted_intent = model_result.intent_type
            predictions[query] = predicted_intent
            print(f"🤖 Modèle  : {predicted_intent} ({model_result.confidence:.2f})")
            print(f"⏱️ Latence : {model_result.processing_time_ms:.1f}ms")

            if predicted_intent == expected_intent:
                print("✅ Match!")
                successes += 1
            else:
                print("❌ Différent")

            latencies.append(model_result.processing_time_ms)
        else:
            predictions[query] = None
            print("⚠️ Mode mock uniquement")

    total = len(test_questions)
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)

    if latencies:
        print(f"\n🎯 PRÉCISION:")
        print(f"   Succès : {successes}/{total} ({(successes/total)*100:.1f}%)")
        print(f"   Échecs : {total - successes}/{total}")

        print(f"\n⏱️ PERFORMANCE:")
        print(f"   Latence moyenne : {sum(latencies)/len(latencies):.1f}ms")
        print(f"   Latence min : {min(latencies):.1f}ms")
        print(f"   Latence max : {max(latencies):.1f}ms")

    print("\n✅ Test terminé!")
    return predictions, successes


def compute_accuracy(predictions, expected):
    """Calcule précision, rappel et F1."""

    total = len(expected)
    true_positive = sum(
        1 for q, p in predictions.items()
        if q in expected and p == expected[q]["intent_type"]
    )

    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# ==================== FONCTION PRINCIPALE ====================

def main():
    """Test amélioré avec meilleure gestion d'erreurs"""
    
    print("=" * 80)
    print("🧪 TEST AMÉLIORÉ - DÉTECTION D'INTENTION PHI-3.5")
    print("=" * 80)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    print("🔧 OPTIONS DE TEST:")
    print("1. Mock uniquement (rapide)")
    print("2. Mock + Modèle (nécessite 4-6GB RAM)")
    print("3. Mock + Modèle avec DEBUG")
    print()
    
    choice = input("Votre choix (1, 2 ou 3) : ").strip()
    use_model = choice in ["2", "3"]
    debug = choice == "3"
    
    print()
    detector = ImprovedIntentDetector(use_model=use_model, debug=debug)

    predictions, successes = evaluate(detector, MOCK_INTENT_RESPONSES)
    metrics = compute_accuracy(predictions, MOCK_INTENT_RESPONSES)

    print(f"\nScore global (F1) : {metrics['f1']:.2f}")

    threshold = 0.8
    return 0 if metrics['f1'] >= threshold else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de détection d'intention")
    parser.add_argument("--use-model", action="store_true", help="Activer le modèle")
    parser.add_argument("--debug", action="store_true", help="Activer le mode débogage")
    args = parser.parse_args()

    try:
        main(use_model=args.use_model, debug=args.debug)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "microsoft/Phi-3.5-mini-instruct"),
        help="Nom ou chemin du modèle à utiliser (peut être local ou sur HuggingFace)",
    )
    args = parser.parse_args()
    try:
        sys.exit(main())

        main(model_name=args.model_name)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
