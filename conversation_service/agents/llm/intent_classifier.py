"""
Intent Classifier - Agent LLM Phase 4
Architecture v2.0 - Composant IA

Responsabilite : Classification autonome des intentions utilisateur
- Few-shot prompting avec exemples contextualises
- Extraction automatique des entites
- Classification avec scores de confiance
- Fallback sur classification par defaut
- Support streaming pour feedback temps reel
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .llm_provider import LLMProviderManager, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class IntentConfidence(Enum):
    """Niveaux de confiance pour la classification"""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.3-0.5
    VERY_LOW = "very_low"  # <0.3

@dataclass
class ClassificationRequest:
    """Requete de classification d'intention"""
    user_message: str
    conversation_context: List[Dict[str, str]]
    user_id: int
    conversation_id: Optional[str] = None
    previous_intent: Optional[str] = None
    use_context: bool = True

@dataclass
class ExtractedEntity:
    """Entite extraite du message utilisateur"""
    name: str
    value: Any
    confidence: float
    span: Tuple[int, int]  # Position dans le texte
    entity_type: str

@dataclass
class ClassificationResult:
    """Resultat de classification d'intention"""
    success: bool
    intent_group: str
    intent_subtype: Optional[str]
    confidence: float
    confidence_level: IntentConfidence
    entities: List[ExtractedEntity]
    reasoning: str
    processing_time_ms: int
    model_used: str
    fallback_used: bool = False
    error_message: Optional[str] = None

class IntentClassifier:
    """
    Agent LLM pour classification autonome d'intentions
    
    Utilise few-shot prompting et extraction d'entites
    """
    
    def __init__(
        self,
        llm_manager: LLMProviderManager,
        few_shot_examples_path: Optional[str] = None
    ):
        self.llm_manager = llm_manager
        self.few_shot_examples_path = few_shot_examples_path
        
        # Service de catégories pour l'arbitrage
        from conversation_service.services.category_service import category_service
        self.category_service = category_service
        
        # Cache des examples few-shot
        self._few_shot_examples: List[Dict] = []
        self._examples_loaded = False
        
        # Configuration intentions supportees
        self.supported_intents = {
            "financial_query": {
                "subtypes": ["balance", "transactions", "expenses", "budget"],
                "entities": ["account_type", "date_range", "amount", "category"],
                "description": "Requetes sur donnees financieres"
            },
            "account_management": {
                "subtypes": ["create", "update", "delete", "list"],
                "entities": ["account_name", "account_type", "parameters"],
                "description": "Gestion des comptes utilisateur"
            },
            "transaction_search": {
                "subtypes": ["simple", "advanced", "filter", "aggregate", "by_period"],
                "entities": ["merchant", "amount_min", "amount_max", "date_start", "date_end", "category", "date_range"],
                "description": "Recherche dans les transactions"
            },
            "CONVERSATIONAL": {
                "subtypes": ["greeting", "help", "goodbye", "clarification"],
                "entities": ["topic", "preference"],
                "description": "Conversation generale"
            }
        }
        
        # Statistiques
        self.stats = {
            "classifications_performed": 0,
            "high_confidence_classifications": 0,
            "fallbacks_used": 0,
            "entity_extractions": 0,
            "avg_confidence": 0.0
        }
        
        logger.info("IntentClassifier initialise")
    
    async def initialize(self) -> bool:
        """Initialise le classificateur et charge les exemples"""
        try:
            # Charger exemples few-shot
            await self._load_few_shot_examples()
            
            logger.info("IntentClassifier initialise avec succes")
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation IntentClassifier: {str(e)}")
            return False
    
    async def classify_intent(self, request: ClassificationRequest) -> ClassificationResult:
        """
        Classifie l'intention d'un message utilisateur
        
        Args:
            request: Requete avec message et contexte
            
        Returns:
            ClassificationResult avec intention et entites
        """
        start_time = datetime.now()
        
        try:
            # 1. Preparation du prompt avec few-shot examples
            system_prompt = self._build_system_prompt()
            classification_prompt = self._build_classification_prompt(request)
            
            # 2. Requete LLM avec few-shot
            llm_request = LLMRequest(
                messages=[{
                    "role": "user",
                    "content": classification_prompt
                }],
                system_prompt=system_prompt,
                few_shot_examples=self._few_shot_examples[:5],  # Top 5 examples
                temperature=0.1,  # Faible pour classification deterministe
                max_tokens=500,
                user_id=request.user_id,
                conversation_id=request.conversation_id
            )
            
            # 3. Generation LLM
            llm_response = await self.llm_manager.generate(llm_request)
            
            if llm_response.error:
                # Fallback sur classification par defaut
                return await self._fallback_classification(request, start_time, llm_response.error)
            
            # 4. Parse de la reponse JSON
            try:
                classification_data = json.loads(llm_response.content.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Erreur parsing JSON classification: {str(e)}")
                return await self._fallback_classification(request, start_time, f"JSON parse error: {str(e)}")
            
            # 5. Validation et construction du resultat
            result = self._build_classification_result(
                classification_data, 
                request, 
                llm_response, 
                start_time
            )
            
            # 6. Mise e jour des statistiques
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur inattendue classification: {str(e)}")
            return await self._fallback_classification(request, start_time, f"Unexpected error: {str(e)}")
    
    def _build_system_prompt(self) -> str:
        """Construit le prompt systeme pour la classification"""
        
        intents_description = "\n".join([
            f"- {intent}: {config['description']}"
            for intent, config in self.supported_intents.items()
        ])
        
        categories_context = self.category_service.build_categories_context()
        
        return f"""Tu es un expert en classification d'intentions pour un assistant financier.

INTENTIONS SUPPORTeES:
{intents_description}

{categories_context}

TeCHE:
1. Analyser le message utilisateur et son contexte
2. Identifier l'intention principale (intent_group) et sous-type (intent_subtype)
3. Extraire toutes les entites pertinentes avec leur position
4. Fournir un score de confiance (0.0 e 1.0)
5. Expliquer ton raisonnement brievement

FORMAT RePONSE (JSON strict):
{{
    "intent_group": "financial_query",
    "intent_subtype": "transactions",
    "confidence": 0.85,
    "entities": [
        {{
            "name": "date_range",
            "value": "last week",
            "confidence": 0.9,
            "span": [15, 25],
            "entity_type": "temporal"
        }}
    ],
    "reasoning": "L'utilisateur demande des informations sur ses transactions recentes"
}}

ReGLES:
- Toujours repondre en JSON valide
- Confidence entre 0.0 et 1.0
- Si incertain, utiliser intent_group="CONVERSATIONAL"
- Extraire maximum 10 entites les plus pertinentes"""
    
    def _build_classification_prompt(self, request: ClassificationRequest) -> str:
        """Construit le prompt de classification avec contexte"""
        
        prompt_parts = []
        
        # Message principal
        prompt_parts.append(f"MESSAGE e CLASSIFIER: \"{request.user_message}\"")
        
        # Contexte conversation si disponible
        if request.use_context and request.conversation_context:
            context_lines = []
            for turn in request.conversation_context[-3:]:  # 3 derniers echanges
                role = turn.get("role", "user")
                content = turn.get("content", "")[:100]  # Limiter longueur
                context_lines.append(f"{role}: {content}")
            
            if context_lines:
                prompt_parts.append(f"CONTEXTE CONVERSATION:\n" + "\n".join(context_lines))
        
        # Intention precedente si disponible
        if request.previous_intent:
            prompt_parts.append(f"INTENTION PReCeDENTE: {request.previous_intent}")
        
        prompt_parts.append("CLASSIFICATION:")
        
        return "\n\n".join(prompt_parts)
    
    def _build_classification_result(
        self, 
        classification_data: Dict[str, Any],
        request: ClassificationRequest,
        llm_response: LLMResponse,
        start_time: datetime
    ) -> ClassificationResult:
        """Construit le resultat de classification e partir des donnees LLM"""
        
        # Validation des champs requis
        intent_group = classification_data.get("intent_group", "CONVERSATIONAL")
        intent_subtype = classification_data.get("intent_subtype")
        confidence = float(classification_data.get("confidence", 0.0))
        reasoning = classification_data.get("reasoning", "No reasoning provided")
        
        # Validation intention supportee
        if intent_group not in self.supported_intents:
            logger.warning(f"Intent non supporte: {intent_group}, fallback vers CONVERSATIONAL")
            intent_group = "CONVERSATIONAL"
            intent_subtype = "clarification"
            confidence *= 0.7  # Reduire confiance
        
        # Parse des entites
        entities = []
        raw_entities = classification_data.get("entities", [])
        
        for entity_data in raw_entities:
            try:
                entity = ExtractedEntity(
                    name=entity_data.get("name", ""),
                    value=entity_data.get("value"),
                    confidence=float(entity_data.get("confidence", 0.0)),
                    span=tuple(entity_data.get("span", [0, 0])),
                    entity_type=entity_data.get("entity_type", "unknown")
                )
                entities.append(entity)
            except (ValueError, TypeError) as e:
                logger.warning(f"Erreur parsing entite: {str(e)}")
                continue
        
        # Determination niveau de confiance
        confidence_level = self._get_confidence_level(confidence)
        
        return ClassificationResult(
            success=True,
            intent_group=intent_group,
            intent_subtype=intent_subtype,
            confidence=confidence,
            confidence_level=confidence_level,
            entities=entities,
            reasoning=reasoning,
            processing_time_ms=self._get_processing_time(start_time),
            model_used=llm_response.model_used,
            fallback_used=False
        )
    
    async def _fallback_classification(
        self, 
        request: ClassificationRequest, 
        start_time: datetime,
        error_message: str
    ) -> ClassificationResult:
        """Classification de fallback en cas d'erreur LLM"""
        
        # Classification heuristique simple
        message_lower = request.user_message.lower()
        
        # Mots-cles pour classification basique
        if any(word in message_lower for word in ["solde", "balance", "compte", "account"]):
            intent_group = "financial_query"
            intent_subtype = "balance"
            confidence = 0.6
        elif any(word in message_lower for word in ["transaction", "achat", "depense", "paiement"]):
            intent_group = "transaction_search"
            intent_subtype = "simple"
            confidence = 0.5
        elif any(word in message_lower for word in ["bonjour", "salut", "hello", "hi"]):
            intent_group = "CONVERSATIONAL"
            intent_subtype = "greeting"
            confidence = 0.8
        elif any(word in message_lower for word in ["aide", "help", "comment"]):
            intent_group = "CONVERSATIONAL"
            intent_subtype = "help"
            confidence = 0.7
        else:
            intent_group = "CONVERSATIONAL"
            intent_subtype = "clarification"
            confidence = 0.3
        
        # Extraction d'entites basique (montants, dates)
        entities = self._extract_basic_entities(request.user_message)
        
        self.stats["fallbacks_used"] += 1
        
        return ClassificationResult(
            success=True,
            intent_group=intent_group,
            intent_subtype=intent_subtype,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            entities=entities,
            reasoning=f"Fallback classification used due to LLM error: {error_message}",
            processing_time_ms=self._get_processing_time(start_time),
            model_used="fallback",
            fallback_used=True,
            error_message=error_message
        )
    
    def _extract_basic_entities(self, message: str) -> List[ExtractedEntity]:
        """Extraction d'entites basique sans LLM"""
        
        entities = []
        message_lower = message.lower()
        
        # Recherche de montants (pattern simple)
        import re
        
        # Montants avec opérateurs en euros
        # Recherche "plus de X euros", "supérieur à X €", etc.
        amount_comparison_patterns = [
            (r'(?:plus de|supérieur(?:e)?s? à|au-dessus de|>\s*)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'gte'),
            (r'(?:moins de|inférieur(?:e)?s? à|en-dessous de|<\s*)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'lt'),
            (r'(?:entre)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:et)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)', 'range')
        ]
        
        for pattern, operator in amount_comparison_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if operator == 'range':
                    # Cas spécial pour "entre X et Y euros"
                    min_amount = float(match.group(1).replace(',', '.'))
                    max_amount = float(match.group(2).replace(',', '.'))
                    entities.append(ExtractedEntity(
                        name="montant",
                        value={"operator": operator, "min": min_amount, "max": max_amount, "currency": "EUR"},
                        confidence=0.85,
                        span=match.span(),
                        entity_type="amount"
                    ))
                else:
                    # Cas standard pour comparaisons simples
                    amount = float(match.group(1).replace(',', '.'))
                    entities.append(ExtractedEntity(
                        name="montant", 
                        value={"operator": operator, "amount": amount, "currency": "EUR"},
                        confidence=0.85,
                        span=match.span(),
                        entity_type="amount"
                    ))
                break  # Prendre seulement le premier match
        
        # Si aucun opérateur trouvé, chercher montant simple
        if not any(entity.name == "montant" for entity in entities):
            amount_pattern = r'(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)'
            match = re.search(amount_pattern, message_lower)
            if match:
                amount = float(match.group(1).replace(',', '.'))
                entities.append(ExtractedEntity(
                    name="montant",
                    value={"operator": "eq", "amount": amount, "currency": "EUR"},
                    confidence=0.7,
                    span=match.span(),
                    entity_type="amount"
                ))
        
        # Dates relatives simples
        date_patterns = [
            (r'hier', 'yesterday'),
            (r'aujourd\'hui', 'today'),
            (r'demain', 'tomorrow'),
            (r'la semaine derniere', 'last_week'),
            (r'le mois dernier', 'last_month'),
            (r'ce mois', 'this_month'),
            (r'cette semaine', 'this_week')
        ]
        
        for pattern, value in date_patterns:
            match = re.search(pattern, message_lower)
            if match:
                entities.append(ExtractedEntity(
                    name="date_range",
                    value=value,
                    confidence=0.7,
                    span=match.span(),
                    entity_type="temporal"
                ))
        
        # Marchands connus (patterns simples)
        merchant_patterns = [
            (r'amazon', 'Amazon'),
            (r'carrefour', 'Carrefour'),
            (r'leclerc', 'Leclerc'),
            (r'auchan', 'Auchan'),
            (r'fnac', 'Fnac'),
            (r'uber', 'Uber'),
            (r'mcdo|mcdonald', 'McDonald\'s'),
            (r'sncf', 'SNCF'),
            (r'total|esso|shell', 'Station Service'),
            (r'restaurant', 'Restaurant'),
            (r'pharmacie', 'Pharmacie')
        ]
        
        for pattern, merchant_name in merchant_patterns:
            match = re.search(pattern, message_lower)
            if match:
                entities.append(ExtractedEntity(
                    name="merchant",
                    value=merchant_name,
                    confidence=0.8,
                    span=match.span(),
                    entity_type="merchant"
                ))
        
        # Catégories de dépenses
        category_patterns = [
            (r'alimentaire|courses|bouffe|nourriture', 'Alimentation'),
            (r'restaurant|resto|café|bar', 'Restaurants'),
            (r'essence|carburant|station', 'Transport'),
            (r'vêtement|fringue|mode', 'Vêtements'),
            (r'santé|médecin|pharmacie', 'Santé'),
            (r'loisir|cinéma|sport', 'Loisirs')
        ]
        
        for pattern, category_name in category_patterns:
            match = re.search(pattern, message_lower)
            if match:
                entities.append(ExtractedEntity(
                    name="category",
                    value=category_name,
                    confidence=0.6,
                    span=match.span(),
                    entity_type="category"
                ))
        
        # Détection du type de transaction basé sur les mots-clés
        transaction_type_patterns = [
            # Mots-clés pour débits (dépenses/sorties)
            (r'(?:dépenses?|depenses?|sorties?|achats?|paiements?|frais|coûts?|couts?)', 'debit'),
            # Mots-clés pour crédits (revenus/entrées)  
            (r'(?:revenus?|gains?|entrées?|entrees?|versements?|salaires?|recettes?|crédits?|credits?)', 'credit'),
        ]
        
        transaction_type_found = False
        for pattern, tx_type in transaction_type_patterns:
            match = re.search(pattern, message_lower)
            if match:
                entities.append(ExtractedEntity(
                    name="transaction_type",
                    value=tx_type,
                    confidence=0.9,
                    span=match.span(),
                    entity_type="transaction_type"
                ))
                transaction_type_found = True
                break
        
        # Si montant détecté mais pas de type explicite, inférer selon le contexte
        if not transaction_type_found and any(entity.name == "montant" for entity in entities):
            # Par défaut, si on parle de montants sans contexte, on assume des dépenses
            # sauf si des mots-clés positifs sont détectés
            if any(word in message_lower for word in ["reçu", "touché", "gagné", "perçu"]):
                entities.append(ExtractedEntity(
                    name="transaction_type",
                    value="credit",
                    confidence=0.6,
                    span=(0, 0),
                    entity_type="transaction_type"
                ))
            else:
                # Inférence par défaut : montants = dépenses
                entities.append(ExtractedEntity(
                    name="transaction_type", 
                    value="debit",
                    confidence=0.7,
                    span=(0, 0),
                    entity_type="transaction_type"
                ))

        return entities[:6]  # Limiter à 6 entités (incluant transaction_type)
    
    def _get_confidence_level(self, confidence: float) -> IntentConfidence:
        """Determine le niveau de confiance"""
        if confidence >= 0.8:
            return IntentConfidence.HIGH
        elif confidence >= 0.5:
            return IntentConfidence.MEDIUM
        elif confidence >= 0.3:
            return IntentConfidence.LOW
        else:
            return IntentConfidence.VERY_LOW
    
    def _update_stats(self, result: ClassificationResult):
        """Met e jour les statistiques de classification"""
        
        self.stats["classifications_performed"] += 1
        self.stats["entity_extractions"] += len(result.entities)
        
        if result.confidence_level == IntentConfidence.HIGH:
            self.stats["high_confidence_classifications"] += 1
        
        # Moyenne mobile de la confiance
        current_avg = self.stats["avg_confidence"]
        total_classifications = self.stats["classifications_performed"]
        
        self.stats["avg_confidence"] = (
            (current_avg * (total_classifications - 1) + result.confidence) / total_classifications
        )
    
    async def _load_few_shot_examples(self) -> None:
        """Charge les exemples few-shot depuis la configuration"""
        
        # Exemples few-shot integres (en attendant fichier de config)
        # PRIORITE: Examples d'opérateurs de montant (positions 1-3 pour être dans top 5)
        self._few_shot_examples = [
            {
                "user": "Mes dépenses de moins de 500 euros", 
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {
            "name": "montant",
            "value": {"operator": "lt", "amount": 500, "currency": "EUR"},
            "confidence": 0.92,
            "span": [4, 33],
            "entity_type": "amount"
        },
        {
            "name": "transaction_type",
            "value": "debit", 
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de dépenses (débit) avec filtre sur montant maximum"
}"""
            },
            {
                "user": "Mes dépenses de plus de 500 euros",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {
            "name": "montant",
            "value": {"operator": "gte", "amount": 500, "currency": "EUR"},
            "confidence": 0.92,
            "span": [4, 32],
            "entity_type": "amount"
        },
        {
            "name": "transaction_type", 
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de dépenses (débit) avec filtre sur montant minimum"
}"""
            },
            {
                "user": "Transactions de 50 euros exactement",
                "assistant": """{
    "intent_group": "transaction_search", 
    "intent_subtype": "by_amount",
    "confidence": 0.88,
    "entities": [
        {
            "name": "montant",
            "value": {"operator": "eq", "amount": 50, "currency": "EUR"},
            "confidence": 0.90,
            "span": [14, 30],
            "entity_type": "amount"
        }
    ],
    "reasoning": "Recherche de transactions avec montant exact"
}"""
            },
            {
                "user": "Mes dépenses du mois de juin",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_period",
    "confidence": 0.90,
    "entities": [
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        },
        {
            "name": "date_range",
            "value": "juin",
            "confidence": 0.90,
            "span": [19, 23],
            "entity_type": "temporal"
        }
    ],
    "reasoning": "Recherche de dépenses par période spécifique (mois)"
}"""
            },
            {
                "user": "Mes achats du mois de mai",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_period",
    "confidence": 0.88,
    "entities": [
        {
            "name": "categories",
            "value": ["Supermarkets / Groceries", "Restaurants", "Clothing", "Electronics", "Online Shopping"],
            "confidence": 0.85,
            "span": [4, 10],
            "entity_type": "categories"
        },
        {
            "name": "date_range",
            "value": "mai",
            "confidence": 0.90,
            "span": [19, 22],
            "entity_type": "temporal"
        }
    ],
    "reasoning": "Recherche d'achats (multiples catégories de dépenses) par période spécifique"
}"""
            },
            {
                "user": "Montre-moi mes achats chez Carrefour la semaine derniere",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "filter",
    "confidence": 0.90,
    "entities": [
        {
            "name": "merchant",
            "value": "Carrefour",
            "confidence": 0.95,
            "span": [20, 29],
            "entity_type": "merchant"
        },
        {
            "name": "date_range",
            "value": "last_week",
            "confidence": 0.85,
            "span": [30, 45],
            "entity_type": "temporal"
        }
    ],
    "reasoning": "Recherche specifique de transactions avec filtre marchand et periode"
}"""
            },
            {
                "user": "Comment puis-je creer un nouveau compte epargne ?",
                "assistant": """{
    "intent_group": "account_management",
    "intent_subtype": "create",
    "confidence": 0.85,
    "entities": [
        {
            "name": "account_type",
            "value": "epargne",
            "confidence": 0.90,
            "span": [35, 42],
            "entity_type": "account_type"
        }
    ],
    "reasoning": "Demande de creation d'un nouveau compte avec type specifie"
}"""
            },
            {
                "user": "Bonjour, j'ai besoin d'aide",
                "assistant": """{
    "intent_group": "CONVERSATIONAL",
    "intent_subtype": "help",
    "confidence": 0.80,
    "entities": [],
    "reasoning": "Salutation avec demande d'assistance generale"
}"""
            },
            {
                "user": "Combien j'ai depense en restaurants ce mois-ci ?",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "aggregate",
    "confidence": 0.88,
    "entities": [
        {
            "name": "category",
            "value": "restaurants",
            "confidence": 0.90,
            "span": [19, 30],
            "entity_type": "category"
        },
        {
            "name": "date_range",
            "value": "this_month",
            "confidence": 0.85,
            "span": [31, 41],
            "entity_type": "temporal"
        }
    ],
    "reasoning": "Demande d'agregation des depenses par categorie et periode"
}"""
            }
        ]
        
        self._examples_loaded = True
        logger.info(f"Charge {len(self._few_shot_examples)} exemples few-shot")
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule le temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    def get_supported_intents(self) -> Dict[str, Any]:
        """Recupere la liste des intentions supportees"""
        return self.supported_intents
    
    def get_stats(self) -> Dict[str, Any]:
        """Recupere les statistiques de classification"""
        return {
            **self.stats,
            "examples_loaded": len(self._few_shot_examples),
            "supported_intents": len(self.supported_intents)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check du classificateur"""
        
        try:
            # Test rapide de classification
            test_request = ClassificationRequest(
                user_message="Test de sante",
                conversation_context=[],
                user_id=0
            )
            
            result = await self.classify_intent(test_request)
            
            return {
                "status": "healthy" if result.success else "degraded",
                "component": "intent_classifier",
                "examples_loaded": len(self._few_shot_examples),
                "supported_intents": len(self.supported_intents),
                "test_classification_success": result.success,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "intent_classifier",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }