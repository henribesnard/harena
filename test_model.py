#!/usr/bin/env python3
"""
Script de test local pour le modèle de détection d'intention Phi-3.5
Utilise les questions du mock Harena pour valider le modèle
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Pour installer les dépendances :
# pip install transformers torch pydantic accelerate sentencepiece protobuf

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("\n📦 Installation des dépendances requises...")
    print("Exécutez : pip install transformers torch pydantic accelerate sentencepiece protobuf")
    sys.exit(1)

# ==================== CONFIGURATION ====================

# Dataset de questions du mock Harena
MOCK_INTENT_RESPONSES = {
    # TRANSACTION_SEARCH
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
    "Historique paiements EDF": {
        "intent_type": "TRANSACTION_SEARCH",
        "intent_category": "FINANCIAL_QUERY", 
        "confidence": 0.89,
        "entities": [
            {"entity_type": "MERCHANT", "raw_value": "EDF", "normalized_value": "edf", "confidence": 0.96}
        ]
    },
    
    # SPENDING_ANALYSIS
    "Analyse dépenses janvier 2024": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.94,
        "entities": [
            {"entity_type": "DATE", "raw_value": "janvier 2024", "normalized_value": "2024-01", "confidence": 0.92}
        ]
    },
    "Où part mon argent ?": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.88
    },
    "Top 5 marchands ce mois": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.90,
        "entities": [
            {"entity_type": "RELATIVE_DATE", "raw_value": "ce mois", "normalized_value": "current_month", "confidence": 0.91}
        ]
    },
    "Évolution budget courses": {
        "intent_type": "BUDGET_TRACKING",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.87,
        "entities": [
            {"entity_type": "CATEGORY", "raw_value": "courses", "normalized_value": "groceries", "confidence": 0.93}
        ]
    },
    "Mes habitudes de dépenses": {
        "intent_type": "SPENDING_ANALYSIS",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.86
    },
    
    # ACCOUNT_BALANCE
    "Quel est mon solde ?": {
        "intent_type": "ACCOUNT_BALANCE",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.96
    },
    "Solde compte courant": {
        "intent_type": "ACCOUNT_BALANCE",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.95,
        "entities": [
            {"entity_type": "ACCOUNT", "raw_value": "compte courant", "normalized_value": "checking", "confidence": 0.94}
        ]
    },
    "Combien sur mon livret A ?": {
        "intent_type": "ACCOUNT_BALANCE",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.93,
        "entities": [
            {"entity_type": "ACCOUNT", "raw_value": "livret A", "normalized_value": "savings_a", "confidence": 0.95}
        ]
    },
    
    # BUDGET_TRACKING
    "Suivi budget alimentation": {
        "intent_type": "BUDGET_TRACKING",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.92,
        "entities": [
            {"entity_type": "CATEGORY", "raw_value": "alimentation", "normalized_value": "food", "confidence": 0.94}
        ]
    },
    "Où en suis-je avec mon budget ?": {
        "intent_type": "BUDGET_TRACKING",
        "intent_category": "FINANCIAL_QUERY",
        "confidence": 0.89
    },
    
    # FILTER_REQUEST
    "Filtre les dépenses > 50€ en janvier": {
        "intent_type": "FILTER_REQUEST",
        "intent_category": "FILTER_REQUEST",
        "confidence": 0.93,
        "entities": [
            {"entity_type": "AMOUNT", "raw_value": "50€", "normalized_value": 50.0, "confidence": 0.96},
            {"entity_type": "DATE", "raw_value": "janvier", "normalized_value": "2025-01", "confidence": 0.90}
        ]
    },
    "Montre seulement les débits": {
        "intent_type": "FILTER_REQUEST",
        "intent_category": "FILTER_REQUEST",
        "confidence": 0.88,
        "entities": [
            {"entity_type": "TRANSACTION_TYPE", "raw_value": "débits", "normalized_value": "debit", "confidence": 0.92}
        ]
    },
    
    # GOAL_TRACKING
    "Objectif épargne 1000€": {
        "intent_type": "GOAL_TRACKING",
        "intent_category": "GOAL_TRACKING",
        "confidence": 0.91,
        "entities": [
            {"entity_type": "AMOUNT", "raw_value": "1000€", "normalized_value": 1000.0, "confidence": 0.97}
        ]
    },
    "Suivi budget vacances 2000€": {
        "intent_type": "GOAL_TRACKING",
        "intent_category": "GOAL_TRACKING",
        "confidence": 0.88,
        "entities": [
            {"entity_type": "CATEGORY", "raw_value": "vacances", "normalized_value": "vacances", "confidence": 0.90},
            {"entity_type": "AMOUNT", "raw_value": "2000€", "normalized_value": 2000.0, "confidence": 0.95}
        ]
    },
    
    # CONVERSATIONAL
    "Bonjour": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "GREETING",
        "confidence": 0.97,
        "entities": []
    },
    "Merci pour l'information": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "CONFIRMATION",
        "confidence": 0.95,
        "entities": []
    },
    "Peux-tu m'expliquer ça plus clairement ?": {
        "intent_type": "CONVERSATIONAL",
        "intent_category": "CLARIFICATION",
        "confidence": 0.92,
        "entities": [],
        "requires_clarification": True
    }
}

# ==================== MODÈLES PYDANTIC ====================

class IntentCategory(str, Enum):
    FINANCIAL_QUERY = "FINANCIAL_QUERY"
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    ACCOUNT_BALANCE = "ACCOUNT_BALANCE"
    BUDGET_TRACKING = "BUDGET_TRACKING"
    GOAL_TRACKING = "GOAL_TRACKING"
    FILTER_REQUEST = "FILTER_REQUEST"
    GREETING = "GREETING"
    CONFIRMATION = "CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    UNKNOWN = "UNKNOWN"

class EntityType(str, Enum):
    AMOUNT = "AMOUNT"
    DATE = "DATE"
    DATE_RANGE = "DATE_RANGE"
    RELATIVE_DATE = "RELATIVE_DATE"
    CATEGORY = "CATEGORY"
    MERCHANT = "MERCHANT"
    ACCOUNT = "ACCOUNT"
    CURRENCY = "CURRENCY"
    TRANSACTION_TYPE = "TRANSACTION_TYPE"

class DetectionMethod(str, Enum):
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"

class FinancialEntity(BaseModel):
    entity_type: EntityType
    raw_value: str
    normalized_value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    detection_method: DetectionMethod = DetectionMethod.LLM_BASED
    validation_status: str = "valid"

class IntentResult(BaseModel):
    intent_type: str = Field(..., min_length=1, max_length=100)
    intent_category: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: List[FinancialEntity] = Field(default_factory=list)
    method: DetectionMethod = Field(default=DetectionMethod.LLM_BASED)
    processing_time_ms: float = Field(..., ge=0.0)
    requires_clarification: bool = False
    search_required: bool = True
    raw_user_message: Optional[str] = None

# ==================== DÉTECTEUR D'INTENTION ====================

class LocalIntentDetector:
    """Détecteur d'intention local utilisant Phi-3.5"""
    
    def __init__(self, use_model: bool = True):
        """
        Args:
            use_model: Si True, charge le modèle. Si False, utilise mock uniquement.
        """
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
        if use_model:
            print("🚀 Chargement du modèle Phi-3.5-mini...")
            try:
                self._load_model()
                print("✅ Modèle chargé avec succès\n")
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle : {e}")
                print("📌 Basculement en mode mock uniquement\n")
                self.use_model = False
    
    def _load_model(self):
        """Charge le modèle Phi-3.5"""
        model_name = "microsoft/Phi-3.5-mini-instruct"
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Modèle (CPU uniquement pour ce test)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
    
    def _create_prompt(self, query: str) -> str:
        """Crée le prompt pour le modèle"""
        return f"""<|system|>
Tu es un agent spécialisé dans l'analyse d'intentions financières.
Analyse la requête et retourne un JSON avec: intent_type, intent_category, confidence, entities.

INTENTIONS: TRANSACTION_SEARCH, SPENDING_ANALYSIS, ACCOUNT_BALANCE, BUDGET_TRACKING, GOAL_TRACKING, CONVERSATIONAL
ENTITÉS: AMOUNT, DATE, CATEGORY, MERCHANT, ACCOUNT

Réponds UNIQUEMENT avec le JSON.
<|end|>
<|user|>
{query}
<|end|>
<|assistant|>"""
    
    def detect_with_model(self, query: str) -> IntentResult:
        """Détection avec le modèle LLM"""
        start = time.time()
        
        try:
            # Tokenisation
            inputs = self.tokenizer(
                self._create_prompt(query),
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Décodage
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            # Parse JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                result_dict = json.loads(response[json_start:json_end])
            else:
                raise ValueError("JSON non trouvé")
            
            # Création du résultat
            entities = []
            for ent in result_dict.get("entities", []):
                entities.append(FinancialEntity(
                    entity_type=EntityType(ent["entity_type"]),
                    raw_value=ent["raw_value"],
                    normalized_value=ent.get("normalized_value", ent["raw_value"]),
                    confidence=ent.get("confidence", 0.9)
                ))
            
            return IntentResult(
                intent_type=result_dict["intent_type"],
                intent_category=IntentCategory(result_dict.get("intent_category", "UNKNOWN")),
                confidence=result_dict.get("confidence", 0.8),
                entities=entities,
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query
            )
            
        except Exception as e:
            # Fallback
            return IntentResult(
                intent_type="ERROR",
                intent_category=IntentCategory.UNKNOWN,
                confidence=0.0,
                entities=[],
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query,
                requires_clarification=True
            )
    
    def detect_with_mock(self, query: str) -> IntentResult:
        """Détection avec les données du mock"""
        start = time.time()
        
        if query in MOCK_INTENT_RESPONSES:
            data = MOCK_INTENT_RESPONSES[query]
            
            # Conversion des entités
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
                raw_user_message=query,
                requires_clarification=data.get("requires_clarification", False)
            )
        else:
            # Question inconnue
            return IntentResult(
                intent_type="UNKNOWN",
                intent_category=IntentCategory.UNKNOWN,
                confidence=0.0,
                entities=[],
                processing_time_ms=(time.time() - start) * 1000,
                raw_user_message=query,
                requires_clarification=True
            )
    
    def detect(self, query: str) -> Tuple[IntentResult, IntentResult]:
        """
        Détecte l'intention avec les deux méthodes
        
        Returns:
            (résultat_mock, résultat_modèle)
        """
        mock_result = self.detect_with_mock(query)
        
        if self.use_model:
            model_result = self.detect_with_model(query)
        else:
            model_result = None
        
        return mock_result, model_result

# ==================== COMPARATEUR DE RÉSULTATS ====================

class ResultComparator:
    """Compare les résultats du mock et du modèle"""
    
    @staticmethod
    def compare_intents(mock: IntentResult, model: IntentResult) -> Dict[str, Any]:
        """Compare deux résultats d'intention"""
        comparison = {
            "intent_match": mock.intent_type == model.intent_type if model else None,
            "category_match": mock.intent_category == model.intent_category if model else None,
            "confidence_diff": abs(mock.confidence - model.confidence) if model else None,
            "entities_comparison": []
        }
        
        if model:
            # Comparaison des entités
            mock_entities = {e.entity_type: e for e in mock.entities}
            model_entities = {e.entity_type: e for e in model.entities}
            
            all_types = set(mock_entities.keys()) | set(model_entities.keys())
            
            for entity_type in all_types:
                mock_ent = mock_entities.get(entity_type)
                model_ent = model_entities.get(entity_type)
                
                comparison["entities_comparison"].append({
                    "type": entity_type,
                    "in_mock": mock_ent is not None,
                    "in_model": model_ent is not None,
                    "values_match": (
                        mock_ent.normalized_value == model_ent.normalized_value
                        if mock_ent and model_ent else None
                    )
                })
        
        return comparison
    
    @staticmethod
    def calculate_accuracy(comparisons: List[Dict]) -> Dict[str, float]:
        """Calcule les métriques de précision"""
        total = len(comparisons)
        if total == 0:
            return {}
        
        intent_matches = sum(1 for c in comparisons if c.get("intent_match") == True)
        category_matches = sum(1 for c in comparisons if c.get("category_match") == True)
        
        return {
            "intent_accuracy": (intent_matches / total) * 100,
            "category_accuracy": (category_matches / total) * 100,
            "total_tests": total
        }

# ==================== TESTS ET AFFICHAGE ====================

def print_header():
    """Affiche l'en-tête du test"""
    print("=" * 80)
    print("🧪 TEST DE DÉTECTION D'INTENTION - DATASET HARENA")
    print("=" * 80)
    print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Questions de test : {len(MOCK_INTENT_RESPONSES)}")
    print("=" * 80)
    print()

def print_test_result(query: str, mock: IntentResult, model: Optional[IntentResult], comparison: Optional[Dict]):
    """Affiche le résultat d'un test"""
    print(f"\n{'─' * 60}")
    print(f"💬 Question : {query}")
    print(f"{'─' * 60}")
    
    # Résultat Mock (attendu)
    print(f"\n📌 RÉSULTAT ATTENDU (Mock):")
    print(f"   Intent : {mock.intent_type}")
    print(f"   Catégorie : {mock.intent_category}")
    print(f"   Confidence : {mock.confidence:.2f}")
    print(f"   Entités : {len(mock.entities)}")
    for ent in mock.entities:
        print(f"      - {ent.entity_type}: '{ent.raw_value}' → {ent.normalized_value}")
    
    # Résultat Modèle
    if model:
        print(f"\n🤖 RÉSULTAT MODÈLE (Phi-3.5):")
        print(f"   Intent : {model.intent_type}")
        print(f"   Catégorie : {model.intent_category}")
        print(f"   Confidence : {model.confidence:.2f}")
        print(f"   Entités : {len(model.entities)}")
        for ent in model.entities:
            print(f"      - {ent.entity_type}: '{ent.raw_value}' → {ent.normalized_value}")
        print(f"   ⏱️ Latence : {model.processing_time_ms:.1f}ms")
        
        # Comparaison
        if comparison:
            print(f"\n📊 COMPARAISON:")
            intent_icon = "✅" if comparison["intent_match"] else "❌"
            cat_icon = "✅" if comparison["category_match"] else "❌"
            print(f"   {intent_icon} Intent : {'Match' if comparison['intent_match'] else 'Différent'}")
            print(f"   {cat_icon} Catégorie : {'Match' if comparison['category_match'] else 'Différent'}")
            if comparison["confidence_diff"] is not None:
                print(f"   📈 Diff confidence : {comparison['confidence_diff']:.2f}")
    else:
        print(f"\n⚠️ Modèle non disponible (mode mock uniquement)")

def print_summary(accuracies: Dict, latencies: List[float]):
    """Affiche le résumé des tests"""
    print(f"\n{'=' * 80}")
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    if accuracies:
        print(f"\n🎯 PRÉCISION:")
        print(f"   Intent : {accuracies['intent_accuracy']:.1f}%")
        print(f"   Catégorie : {accuracies['category_accuracy']:.1f}%")
        print(f"   Tests : {accuracies['total_tests']}")
    
    if latencies:
        print(f"\n⏱️ PERFORMANCE:")
        print(f"   Latence moyenne : {sum(latencies)/len(latencies):.1f}ms")
        print(f"   Latence min : {min(latencies):.1f}ms")
        print(f"   Latence max : {max(latencies):.1f}ms")
    
    print("\n" + "=" * 80)

# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale de test"""
    print_header()
    
    # Demander le mode
    print("🔧 MODE DE TEST:")
    print("1. Mock uniquement (rapide)")
    print("2. Mock + Modèle LLM (nécessite 4-6GB RAM)")
    print()
    
    choice = input("Votre choix (1 ou 2) : ").strip()
    use_model = (choice == "2")
    
    print()
    detector = LocalIntentDetector(use_model=use_model)
    comparator = ResultComparator()
    
    # Variables pour statistiques
    comparisons = []
    latencies = []
    
    # Test sur toutes les questions
    print("\n🚀 DÉBUT DES TESTS\n")
    
    for i, query in enumerate(MOCK_INTENT_RESPONSES.keys(), 1):
        print(f"\n[{i}/{len(MOCK_INTENT_RESPONSES)}] Test en cours...")
        
        # Détection
        mock_result, model_result = detector.detect(query)
        
        # Comparaison
        if model_result:
            comparison = comparator.compare_intents(mock_result, model_result)
            comparisons.append(comparison)
            latencies.append(model_result.processing_time_ms)
        else:
            comparison = None
        
        # Affichage
        print_test_result(query, mock_result, model_result, comparison)
        
        # Pause entre tests pour ne pas surcharger
        if use_model and i < len(MOCK_INTENT_RESPONSES):
            time.sleep(0.1)
    
    # Calcul et affichage des statistiques
    if comparisons:
        accuracies = comparator.calculate_accuracy(comparisons)
    else:
        accuracies = {}
    
    print_summary(accuracies, latencies)
    
    print("\n✅ Tests terminés !")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()