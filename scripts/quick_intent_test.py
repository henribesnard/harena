"""
Agent OpenAI spécialisé pour la détection d'intention financière Harena
Utilise les Structured Outputs et l'optimisation des coûts via Batch API
"""

import json
import time
import asyncio
import os
from typing import Dict, List, Any, Optional, Tuple, Literal
from pathlib import Path
from datetime import datetime
import openai
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import logging

from conversation_service.models.financial_models import (
    IntentCategory,
    EntityType,
    FinancialEntity,
    IntentResult,
    DetectionMethod,
)
from conversation_service.utils.validators import validate_intent_result_contract

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping to harmonize categories between INTENTS.md and the agent
CATEGORY_MAP = {
    "ACCOUNT_BALANCE": "BALANCE_INQUIRY",
    "GENERAL_QUESTION": "UNCLEAR_INTENT",
}

# ==================== MODÈLES PYDANTIC (IntentResult) ====================

class IntentCategory(str, Enum):
    """Catégories d'intentions consultatives."""
    FINANCIAL_QUERY = "FINANCIAL_QUERY"
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    BALANCE_INQUIRY = "BALANCE_INQUIRY"
    ACCOUNT_INFORMATION = "ACCOUNT_INFORMATION"

    SPENDING_ANALYSIS = "SPENDING_ANALYSIS"
    BUDGET_ANALYSIS = "BUDGET_ANALYSIS"
    TREND_ANALYSIS = "TREND_ANALYSIS"

    GREETING = "GREETING"
    CLARIFICATION = "CLARIFICATION"
    CONFIRMATION = "CONFIRMATION"
    GENERAL_QUESTION = "GENERAL_QUESTION"

    EXPORT_REQUEST = "EXPORT_REQUEST"
    FILTER_REQUEST = "FILTER_REQUEST"
    SORT_REQUEST = "SORT_REQUEST"
    GOAL_TRACKING = "GOAL_TRACKING"

    UNCLEAR_INTENT = "UNCLEAR_INTENT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"

    # Valeur supplémentaire spécifique au script
    UNKNOWN = "UNKNOWN"

class EntityType(str, Enum):
    """Types d'entités financières."""
    # Monetary entities
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"

    # Temporal entities
    DATE = "DATE"
    DATE_RANGE = "DATE_RANGE"
    RELATIVE_DATE = "RELATIVE_DATE"

    # Account entities
    ACCOUNT_TYPE = "ACCOUNT_TYPE"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    IBAN = "IBAN"

    # Transaction entities
    TRANSACTION_TYPE = "TRANSACTION_TYPE"
    OPERATION_TYPE = "OPERATION_TYPE"
    MERCHANT = "MERCHANT"
    CATEGORY = "CATEGORY"
    DESCRIPTION = "DESCRIPTION"

    # Person/Organization entities
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    BANK = "BANK"

    # Location entities
    LOCATION = "LOCATION"
    COUNTRY = "COUNTRY"

    # Financial instruments
    CARD_TYPE = "CARD_TYPE"
    PAYMENT_METHOD = "PAYMENT_METHOD"

    # Other
    REFERENCE_NUMBER = "REFERENCE_NUMBER"
    OTHER = "OTHER"

class DetectionMethod(str, Enum):
    """Méthode de détection."""
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"

class FinancialEntity(BaseModel):
    """Entité financière extraite."""
    entity_type: EntityType
    raw_value: str
    normalized_value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    start_position: Optional[int] = Field(default=None, ge=0)
    end_position: Optional[int] = Field(default=None, ge=0)
    detection_method: DetectionMethod = DetectionMethod.LLM_BASED
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validation_status: Literal["valid", "invalid", "uncertain", "not_validated"] = "not_validated"
    alternative_values: Optional[List[Any]] = None

    @model_validator(mode='after')
    def validate_positions(self) -> 'FinancialEntity':
        if (self.start_position is not None and
                self.end_position is not None and
                self.end_position <= self.start_position):
            raise ValueError("end_position must be greater than start_position")
        return self

    def to_search_filter(self) -> Optional[Dict[str, Any]]:
        """Convertit l'entité en filtre de recherche."""
        if self.validation_status == "invalid":
            return None
        filter_map = {
            EntityType.AMOUNT: {"field": "amount", "value": self.normalized_value},
            EntityType.DATE: {"field": "date", "value": self.normalized_value},
            EntityType.DATE_RANGE: {"field": "date", "value": self.normalized_value},
            EntityType.CATEGORY: {"field": "category", "value": self.normalized_value},
            EntityType.MERCHANT: {"field": "merchant", "value": self.normalized_value},
            EntityType.TRANSACTION_TYPE: {"field": "transaction_type", "value": self.normalized_value},
            EntityType.OPERATION_TYPE: {"field": "operation_type", "value": self.normalized_value},
        }
        return filter_map.get(self.entity_type)

class IntentResult(BaseModel):
    """Résultat complet de classification d'intention."""
    intent_type: str = Field(..., min_length=1, max_length=100)
    intent_category: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: List[FinancialEntity] = Field(default_factory=list)
    method: DetectionMethod = Field(default=DetectionMethod.LLM_BASED)
    processing_time_ms: float = Field(..., ge=0.0)
    alternative_intents: Optional[List[Dict[str, Any]]] = None
    context_influence: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[str]] = None
    requires_clarification: bool = False
    suggested_actions: Optional[List[str]] = None
    raw_user_message: Optional[str] = None
    normalized_query: Optional[str] = None
    search_required: bool = True

    @field_validator("alternative_intents")
    @classmethod
    def validate_alternative_intents(cls, v):
        if v is not None:
            for alt in v:
                if "intent_type" not in alt or "confidence" not in alt:
                    raise ValueError("Alternative intent mal formé")
                if not (0.0 <= alt["confidence"] <= 1.0):
                    raise ValueError("Confidence doit être entre 0 et 1")
        return v

    @model_validator(mode='after')
    def validate_entities_consistency(self):
        """Valide la cohérence des entités avec l'intention."""
        return self

# Importés depuis conversation_service.models.financial_models

# ==================== AGENT OPENAI OPTIMISÉ ====================

class HarenaIntentAgent:
    """
    Agent OpenAI spécialisé pour la détection d'intention Harena.
    Utilise les Structured Outputs pour garantir le format IntentResult.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",  # Modèle économique par défaut
        use_batch_api: bool = False,
        cache_enabled: bool = True
    ):
        """
        Initialise l'agent avec configuration optimisée.
        
        Args:
            api_key: Clé API OpenAI
            model: Modèle à utiliser (gpt-4o-mini recommandé pour coût)
            use_batch_api: Utiliser Batch API pour 50% de réduction
            cache_enabled: Activer le cache pour économies
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.use_batch_api = use_batch_api
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        
        # Configuration des coûts (par million de tokens)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60, "batch_discount": 0.5},
            "gpt-4o": {"input": 2.50, "output": 10.00, "batch_discount": 0.5},
            "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00, "batch_discount": 0.5}
        }
        
        # Prompt système optimisé
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"Agent initialisé avec modèle {model}")
        if use_batch_api:
            logger.info("Mode Batch API activé (50% de réduction)")
    
    def _create_system_prompt(self) -> str:
        """Crée le prompt système avec la liste complète des intentions et exemples."""

        intents: List[Tuple[str, str, str]] = []
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "INTENTS.md"
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("|") or line.startswith("| ---") or "Intent Type" in line:
                    continue
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) < 3:
                    continue
                intent, category, description = parts[:3]
                if category.startswith("UNSUPPORTED"):
                    category = "UNCLEAR_INTENT"
                category = CATEGORY_MAP.get(category, category)
                intents.append((intent, category, description))

        example_queries: Dict[str, str] = {
            "TRANSACTION_SEARCH": "Liste toutes mes transactions",
            "SEARCH_BY_DATE": "Transactions de mars 2024",
            "SEARCH_BY_AMOUNT": "Transactions de 50 euros",
            "SEARCH_BY_MERCHANT": "Transactions chez Carrefour",
            "SEARCH_BY_CATEGORY": "Transactions de la catégorie restaurants",
            "SEARCH_BY_AMOUNT_AND_DATE": "Achats de plus de 100 euros en janvier 2024",
            "SEARCH_BY_OPERATION_TYPE": "Transactions par carte bancaire",
            "SEARCH_BY_TEXT": "Recherche les transactions contenant Netflix",
            "COUNT_TRANSACTIONS": "Combien de transactions en février ?",
            "MERCHANT_INQUIRY": "Analyse des dépenses chez Amazon",
            "FILTER_REQUEST": "Seulement les débits pas les crédits",
            "SPENDING_ANALYSIS": "Analyse de mes dépenses",
            "SPENDING_ANALYSIS_BY_CATEGORY": "Analyse des dépenses alimentaires",
            "SPENDING_ANALYSIS_BY_PERIOD": "Analyse de mes dépenses la semaine dernière",
            "SPENDING_COMPARISON": "Compare mes dépenses de janvier et février",
            "TREND_ANALYSIS": "Tendance de mes dépenses cette année",
            "CATEGORY_ANALYSIS": "Répartition de mes dépenses par catégorie",
            "COMPARISON_QUERY": "Compare restaurants et courses",
            "BALANCE_INQUIRY": "Quel est mon solde actuel ?",
            "ACCOUNT_BALANCE_SPECIFIC": "Solde de mon compte courant",
            "BALANCE_EVOLUTION": "Évolution de mon solde",
            "GREETING": "Bonjour",
            "CONFIRMATION": "Merci beaucoup",
            "CLARIFICATION": "Peux-tu préciser ?",
            "GENERAL_QUESTION": "Quel temps fait-il ?",
            "TRANSFER_REQUEST": "Transfère 100 euros à Paul",
            "PAYMENT_REQUEST": "Paye ma facture d'électricité",
            "CARD_BLOCK": "Bloque ma carte bancaire",
            "BUDGET_INQUIRY": "Où en est mon budget ?",
            "GOAL_TRACKING": "Quel est l'état de mon objectif d'épargne ?",
            "EXPORT_REQUEST": "Exporte mes transactions en CSV",
            "OUT_OF_SCOPE": "Donne-moi une recette de cuisine",
            "UNCLEAR_INTENT": "Je ne sais pas fais quelque chose",
            "UNKNOWN": "blabla ???",
            "TEST_INTENT": "[TEST] ping",
            "ERROR": "��",
        }

        unsupported = {
            "TRANSFER_REQUEST",
            "PAYMENT_REQUEST",
            "CARD_BLOCK",
            "BUDGET_INQUIRY",
            "GOAL_TRACKING",
            "EXPORT_REQUEST",
            "OUT_OF_SCOPE",
        }

        lines = [
            "Tu es un agent spécialisé dans l'analyse d'intentions financières pour le système Harena.",
            "Tu dois analyser chaque question et retourner un objet IntentResult structuré.",
            "",
            "INTENTIONS DÉTAILLÉES ET EXEMPLES:",
        ]

        for intent, category, description in intents:
            user_example = example_queries.get(intent, "")
            example_intent = "UNSUPPORTED" if intent in unsupported else intent
            lines.append(f"- {intent} ({category}): {description}")
            if user_example:
                lines.append(
                    f"  Ex: \"{user_example}\" -> intent_type={example_intent}, intent_category={category}, entities=[]"
                )

        lines += [
            "",
            "ENTITÉS À EXTRAIRE:",
            "- AMOUNT: Montants (50€, mille euros) → normaliser en float",
            "- DATE/DATE_RANGE: Dates (janvier 2024) → format ISO",
            "- MERCHANT: Marchands (Carrefour, Netflix) → lowercase",
            "- CATEGORY: Catégories (alimentation, transport) → termes canoniques",
            "- ACCOUNT: Comptes (livret A, compte courant) → identifiants standards",
            "",
            "RÈGLES IMPORTANTES:",
            "1. Toujours retourner un IntentResult complet avec tous les champs",
            "2. Normaliser les valeurs (montants en float, dates en ISO, marchands en lowercase)",
            "3. Confidence entre 0.80 et 0.99 selon la clarté",
            "4. Processing_time_ms réaliste (100-300ms)",
            "5. Suggested_actions pertinentes pour l'intention",
            "6. Si l'utilisateur demande une action (paiement, virement, transfert, export),",
            '   répondre avec intent_type="UNSUPPORTED" et intent_category="UNCLEAR_INTENT".',
            "   Ajoute une note de clarification indiquant que cette action n'est pas supportée.",
            "",
            'Exemples: "Combien j\'ai dépensé chez Carrefour ?" → SPENDING_ANALYSIS avec MERCHANT:carrefour',
            '"Quel est mon solde ?" → BALANCE_INQUIRY avec search_required:true',
        ]

        return "\n".join(lines)
    
    def _get_json_schema(self) -> dict:
        """
        Retourne le JSON Schema pour les Structured Outputs.
        Compatible avec l'API OpenAI.
        """
        intent_categories = [cat.value for cat in IntentCategory]
        return {
            "name": "intent_result",
            "schema": {
                "type": "object",
                "properties": {
                    "intent_type": {
                        "type": "string",
                        "description": "Type d'intention détecté"
                    },
                    "intent_category": {
                        "type": "string",
                        "enum": intent_categories,
                        "description": "Catégorie d'intention"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Score de confiance"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_type": {
                                    "type": "string",
                                    "enum": [et.value for et in EntityType]
                                },
                                "raw_value": {"type": "string"},
                                "normalized_value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "object", "additionalProperties": False},

                                        {
                                            "type": "object",
                                            "properties": {
                                                "start_date": {"type": "string"},
                                                "end_date": {"type": "string"}
                                            },
                                            "required": ["start_date", "end_date"],
                                            "additionalProperties": False
                                        },
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "number"},
                                                    {"type": "boolean"},
                                                    {"type": "object", "properties": {}, "additionalProperties": False},
                                                    {"type": "null"}
                                                ]
                                            }
                                        },
                                        {"type": "boolean"},
                                        {"type": "null"}
                                    ]
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "detection_method": {
                                    "type": "string",
                                    "enum": ["llm_based", "rule_based", "hybrid"]
                                },
                                "validation_status": {"type": "string"}
                            },
                            "required": ["entity_type", "raw_value", "normalized_value", 
                                       "confidence", "detection_method", "validation_status"],
                            "additionalProperties": False
                        },
                        "description": "Entités extraites"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["llm_based", "rule_based", "hybrid"]
                    },
                    "processing_time_ms": {
                        "type": "number",
                        "minimum": 0.0
                    },
                    "requires_clarification": {"type": "boolean"},
                    "search_required": {"type": "boolean"},
                    "suggested_actions": {
                        "type": ["array", "null"],
                        "items": {"type": "string"}
                    },
                    "raw_user_message": {"type": ["string", "null"]},
                    "normalized_query": {"type": ["string", "null"]},
                    "alternative_intents": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "intent_type": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["intent_type", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "validation_errors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"}
                    },
                    "context_influence": {
                        "type": ["object", "null"],
                        "properties": {},
                        "additionalProperties": False
                    }
                },
                "required": [
                    "intent_type", "intent_category", "confidence", "entities",
                    "method", "processing_time_ms", "requires_clarification",
                    "search_required", "suggested_actions", "raw_user_message",
                    "normalized_query", "alternative_intents", "validation_errors",
                    "context_influence"
                ],
                "additionalProperties": False
            }
        }
    
    async def detect_intent_async(self, user_message: str) -> IntentResult:
        """
        Détection d'intention asynchrone avec Structured Outputs.
        
        Args:
            user_message: Question de l'utilisateur
            
        Returns:
            IntentResult structuré et validé
        """
        start_time = time.time()
        
        # Vérifier le cache
        if self.cache_enabled and user_message in self.cache:
            cached_result = self.cache[user_message].copy()
            cached_result["processing_time_ms"] = 0.5  # Cache hit
            result = IntentResult(**cached_result)
            errors = validate_intent_result_contract(result.model_dump())
            if errors:
                logger.error(f"IntentResult contract errors: {errors}")
                result.validation_errors = errors
            return result
        
        try:
            # Appel API avec Structured Outputs
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyse cette question et retourne un IntentResult: {user_message}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": self._get_json_schema()
                },
                temperature=0.1,  # Déterministe
                max_tokens=500,
                seed=42  # Pour reproductibilité
            )
            
            # Parser la réponse JSON
            result_dict = json.loads(response.choices[0].message.content)
            
            # Ajouter métadonnées
            processing_time = (time.time() - start_time) * 1000
            result_dict["processing_time_ms"] = processing_time
            result_dict["raw_user_message"] = user_message

            # Valider avec Pydantic
            result = IntentResult(**result_dict)

            # Validation du contrat
            errors = validate_intent_result_contract(result.model_dump())
            if errors:
                logger.error(f"IntentResult contract errors: {errors}")
                result.validation_errors = (result.validation_errors or []) + errors

            # Post-traitement pour intents non supportés
            if result.intent_type == "UNSUPPORTED":
                result.validation_errors = ["Intent not supported."]
                result.search_required = False
                result.requires_clarification = True

            # Mettre en cache si succès
            if self.cache_enabled and result.confidence > 0.8:
                self.cache[user_message] = result.model_dump()
            
            # Log des coûts estimés
            self._log_cost_estimate(response.usage)

            return result

        except Exception as e:
            logger.error(f"Erreur détection: {e}")
            # Retour d'erreur structuré
            result = IntentResult(
                intent_type="ERROR",
                intent_category=IntentCategory.UNCLEAR_INTENT,
                confidence=0.0,
                entities=[],
                method=DetectionMethod.LLM_BASED,
                processing_time_ms=(time.time() - start_time) * 1000,
                raw_user_message=user_message,
                validation_errors=[str(e)],
                requires_clarification=True,
                search_required=False,
            )
            errors = validate_intent_result_contract(result.model_dump())
            if errors:
                logger.error(f"IntentResult contract errors: {errors}")
                result.validation_errors = (result.validation_errors or []) + errors
            return result
    
    def detect_intent(self, user_message: str) -> IntentResult:
        """Version synchrone de detect_intent."""
        return asyncio.run(self.detect_intent_async(user_message))
    
    async def batch_detect_intents(self, messages: List[str]) -> List[IntentResult]:
        """
        Traitement batch pour économiser 50% des coûts.
        
        Args:
            messages: Liste de questions à traiter
            
        Returns:
            Liste de résultats IntentResult
        """
        if not self.use_batch_api:
            # Traitement parallèle standard
            tasks = [self.detect_intent_async(msg) for msg in messages]
            return await asyncio.gather(*tasks)
        
        # Créer le batch pour l'API
        batch_requests = []
        for i, message in enumerate(messages):
            batch_requests.append({
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Analyse: {message}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": self._get_json_schema()
                    },
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            })
        
        # Créer le fichier batch
        batch_file_content = "\n".join(json.dumps(req) for req in batch_requests)
        
        # Upload et créer le batch
        file = self.client.files.create(
            file=batch_file_content.encode(),
            purpose="batch"
        )
        
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        logger.info(f"Batch créé: {batch.id} - Économie de 50% sur {len(messages)} requêtes")
        
        # Note: En production, implémenter le polling du statut du batch
        # et récupérer les résultats une fois terminé
        
        return []  # Placeholder - implémenter récupération async
    
    def _log_cost_estimate(self, usage: Any):
        """Log l'estimation des coûts."""
        if usage and self.model in self.pricing:
            prices = self.pricing[self.model]
            input_cost = (usage.prompt_tokens / 1_000_000) * prices["input"]
            output_cost = (usage.completion_tokens / 1_000_000) * prices["output"]
            
            if self.use_batch_api:
                input_cost *= prices["batch_discount"]
                output_cost *= prices["batch_discount"]
            
            total_cost = input_cost + output_cost
            
            logger.debug(f"Coût estimé: ${total_cost:.6f} "
                        f"(input: {usage.prompt_tokens} tokens, "
                        f"output: {usage.completion_tokens} tokens)")
    
    def optimize_for_production(self):
        """Optimisations pour la production."""
        optimizations = {
            "model": "gpt-4o-mini",  # Plus économique
            "batch_api": True,  # 50% de réduction
            "cache": True,  # Éviter requêtes répétées
            "prompt_caching": True,  # 75% réduction sur prompts répétés
            "strategies": [
                "Utiliser gpt-4o-mini : 0.15$/1M input, 0.60$/1M output",
                "Batch API : -50% sur tous les coûts",
                "Prompt caching : -75% sur tokens système répétés",
                "Cache local : 0$ pour requêtes identiques",
                "Structured Outputs : Évite retry et parsing errors"
            ]
        }
        return optimizations

# ==================== UTILISATION ET TESTS ====================

async def main():
    """Exemple d'utilisation de l'agent."""
    
    # Charger les variables d'environnement
    import os
    from dotenv import load_dotenv
    
    load_dotenv()  # Charge le fichier .env
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Erreur : OPENAI_API_KEY non trouvée dans .env")
        print("Créez un fichier .env avec : OPENAI_API_KEY=sk-...")
        return
    
    # Initialiser l'agent
    agent = HarenaIntentAgent(
        api_key=api_key,
        model="gpt-4o-mini",  # Modèle économique
        use_batch_api=False,  # True pour 50% de réduction
        cache_enabled=True
    )
    
    # Tests unitaires
    test_queries = [
        "Combien j'ai dépensé chez Carrefour le mois dernier ?",
        "Combien ai-je transféré à Marie le mois dernier ?",

        "Quel est le solde de mon livret A ?",
        "Montre mes achats supérieurs à 100€ en janvier 2024",
        "Budget alimentation ce mois",
        "Bonjour, comment ça va ?",
        "Merci pour l'information",
        "Peux-tu transférer 500 euros à Marie ?",  # Doit renvoyer une intention non supportée
    ]
    
    print("\n🚀 TEST DE L'AGENT HARENA INTENT\n")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\n💬 Question : {query}")
        
        # Détection d'intention
        result = await agent.detect_intent_async(query)
        
        print(f"📌 Intent : {result.intent_type}")
        print(f"📊 Catégorie : {result.intent_category}")
        print(f"🎯 Confidence : {result.confidence:.2%}")
        print(f"⏱️ Latence : {result.processing_time_ms:.1f}ms")

        if result.entities:
            print(f"📝 Entités :")
            for entity in result.entities:
                print(f"   - {entity.entity_type}: '{entity.raw_value}' → {entity.normalized_value}")

        if result.suggested_actions:
            print(f"💡 Actions : {', '.join(result.suggested_actions)}")

        if query == "Peux-tu transférer 500 euros à Marie ?":
            assert (
                result.intent_type in {"UNSUPPORTED", "UNCLEAR", "UNCLEAR_INTENT"}
                or result.intent_category in {IntentCategory.UNCLEAR_INTENT, IntentCategory.OUT_OF_SCOPE}
            ), "Les demandes de paiement doivent être marquées comme non supportées ou ambiguës"

        print("-" * 60)
    
    # Afficher les optimisations
    print("\n💰 OPTIMISATIONS DE COÛT :")
    optimizations = agent.optimize_for_production()
    for strategy in optimizations["strategies"]:
        print(f"  ✅ {strategy}")
    
    print("\n📊 ESTIMATION MENSUELLE (10 000 requêtes/jour) :")
    daily_requests = 10_000
    tokens_per_request = 200  # Moyenne estimée
    
    # Calcul sans optimisation
    monthly_cost_standard = (daily_requests * 30 * tokens_per_request / 1_000_000) * (0.15 + 0.60)
    print(f"  Sans optimisation : ${monthly_cost_standard:.2f}/mois")
    
    # Calcul avec Batch API
    monthly_cost_batch = monthly_cost_standard * 0.5
    print(f"  Avec Batch API : ${monthly_cost_batch:.2f}/mois")
    
    # Calcul avec cache (30% de requêtes en cache)
    monthly_cost_cached = monthly_cost_batch * 0.7
    print(f"  Avec Batch + Cache : ${monthly_cost_cached:.2f}/mois")

if __name__ == "__main__":
    asyncio.run(main())
