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
                "subtypes": ["simple", "advanced", "filter", "aggregate", "by_period", "by_category", "by_date", "by_merchant", "by_amount", "by_type_and_date"],
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
            
            # 2. Requete LLM avec few-shot ET JSON OUTPUT NATIF DeepSeek
            llm_request = LLMRequest(
                messages=[{
                    "role": "user",
                    "content": classification_prompt
                }],
                system_prompt=system_prompt,
                few_shot_examples=self._few_shot_examples[:5],  # Top 5 examples
                temperature=0.1,  # Faible pour classification deterministe
                max_tokens=1000,  # Plus de tokens pour les entités complètes
                response_format={"type": "json_object"},  # FORCER JSON OUTPUT natif DeepSeek
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
        
        return f"""Tu es un agent LLM expert en classification d'intentions et extraction d'entités pour un assistant financier personnel.

🚨 RÈGLE PRIORITAIRE ABSOLUE : "categories" EST BANNI 🚨
- NE JAMAIS utiliser "categories" dans les réponses JSON
- TOUJOURS utiliser "query" pour les recherches textuelles
- INTERDICTION TOTALE de categories: [...]

=== INTENTIONS SUPPORTÉES ===
{intents_description}

=== LOGIQUE D'EXTRACTION D'ENTITÉS ===

🚨 RÈGLE CRITIQUE: STRUCTURE entities_structured OBLIGATOIRE 🚨
- Toujours inclure transaction_type même si non explicite
- Utiliser les opérateurs standardisés: gt, lt, gte, lte, eq
- Structure cohérente: simple (amount+operator) OU plage (amount_min+amount_max)

1. MONTANTS ET OPÉRATEURS :
   - "plus de 100 euros" → amount: 100, operator: "gt", transaction_type: "debit"
   - "moins de 50€" → amount: 50, operator: "lt", transaction_type: "debit"
   - "exactement 200 euros" → amount: 200, operator: "eq", transaction_type: "debit"
   - "entre 50 et 100 euros" → amount_min: 50, amount_max: 100, transaction_type: "all"
   - "au moins 500€" → amount: 500, operator: "gte", transaction_type: "debit"
   - "maximum 1000 euros" → amount: 1000, operator: "lte", transaction_type: "all"
   - "500 euros ou plus" → amount: 500, operator: "gte", transaction_type: "all"
   - "jusqu'à 300€" → amount: 300, operator: "lte", transaction_type: "all"
   - "à partir de 150 euros" → amount: 150, operator: "gte", transaction_type: "all"

2. DATES ET PÉRIODES (FORMAT date_range UNIQUEMENT) :

   🗓️ DATES RELATIVES (priorité haute) :
   - "cette semaine" → date_range: "this_week"
   - "le mois dernier" → date_range: "last_month"
   - "aujourd'hui" → date_range: "today"
   - "demain" → date_range: "tomorrow"
   - "hier" → date_range: "yesterday"
   - "cette année" → date_range: "this_year"
   - "l'année dernière" → date_range: "last_year"
   - "du weekend" → date_range: "weekend"
   - "ce mois" → date_range: "this_month"
   - "des 30 derniers jours" → date_range: "last_30_days"

   📅 MOIS SPÉCIFIQUES :
   - "de mai" → date_range: "2025-05"
   - "en janvier 2025" → date_range: "2025-01"
   - "en décembre" → date_range: "2025-12"
   - "d'octobre" → date_range: "2025-10"

   📍 DATES SPÉCIFIQUES (problématiques - renforcées) :
   - "du 1er mai" → date_range: "2025-05-01"
   - "du 5 mars" → date_range: "2025-03-05"
   - "du 15 septembre" → date_range: "2025-09-15"
   - "du 20 juin" → date_range: "2025-06-20"
   - "du 31 décembre" → date_range: "2025-12-31"

   📊 PLAGES DE DATES :
   - "du 14 au 15 mai" → date_range: "2025-05-14_2025-05-15"
   - "du 1er au 15 octobre" → date_range: "2025-10-01_2025-10-15"
   - "du 10 au 20 mars" → date_range: "2025-03-10_2025-03-20"

   🎯 ANNÉES SPÉCIFIQUES :
   - "de 1995" → date_range: "1995"
   - "en 2030" → date_range: "2030"
   - "d'avril 2024" → date_range: "2024-04"

   ⚠️ RÈGLES CRITIQUES :
   - TOUJOURS utiliser 'date_range' - JAMAIS month, year, date_specific
   - Format strict: YYYY-MM-DD, YYYY-MM, YYYY
   - Dates françaises : "1er" = "01", "5" = "05"
   - Année par défaut 2025 sauf si spécifiée
   - Plages avec underscore : "YYYY-MM-DD_YYYY-MM-DD"

3. MARCHANDS ET COMMERÇANTS :
   - UN SEUL marchand : "Mes achats Tesla" → merchant: "Tesla", transaction_type: "debit"
   - PLUSIEURS marchands : "Amazon Prime Video Netflix Disney+" → merchants: ["Amazon Prime Video", "Netflix", "Disney+"], transaction_type: "debit"
   - Corriger automatiquement les fautes de frappe : "Netflik" → "Netflix", "Amazone" → "Amazon"
   - Détecter les marques connues : "Tesla", "Amazon", "McDonald's", "Uber", "Google"
   - Normaliser : "mcdo" → "McDonald's", "Apple/iTunes" → "Apple/iTunes"
   - OBLIGATOIRE: Toujours ajouter transaction_type même pour marchands

4. CATÉGORIES → PRIORITÉ AUX CATÉGORIES DE BASE :
   🎯 RÈGLE CORRIGÉE : Utiliser "categories" pour les catégories EXISTANTES en base, "query" pour le reste

   🏆 CATÉGORIES DE BASE DISPONIBLES (utiliser categories):
   - "dépenses restaurant" → categories: ["Restaurants"], transaction_type: "debit"
   - "frais de transport" → categories: ["Public Transportation", "Taxi/Uber", "Fuel", "Car Maintenance", "Parking"], transaction_type: "debit"
   - "achats alimentaires" → categories: ["Supermarkets / Groceries", "Restaurants", "Fast foods", "Coffee shop", "Food - Others"], transaction_type: "debit"
   - "dépenses santé" → categories: ["Doctor Visits", "Dentist", "Pharmacy", "Medical Equipment", "Medical Insurance"], transaction_type: "debit"
   - "sorties loisirs" → categories: ["Movies & Cinema", "Concerts & Shows", "Gaming", "Sports Events", "Streaming Services"], transaction_type: "debit"
   - "factures d'énergie" → categories: ["Electricity", "Water"], transaction_type: "debit"
   - "frais bancaires" → categories: ["Bank Fees"], transaction_type: "debit"
   - "achats vêtements" → categories: ["Clothing"], transaction_type: "debit"
   - "dépenses électronique" → categories: ["Electronics"], transaction_type: "debit"
   - "achats en ligne" → categories: ["Online Shopping"], transaction_type: "debit"
   - "abonnements" → categories: ["Streaming Services", "Internet/Phone"], transaction_type: "debit"

   🔎 CAS NÉCESSITANT QUERY (pas de catégorie en base):
   - "dépenses spatial" → query: "spatial espace astronomie", transaction_type: "debit"
   - "achats Bitcoin" → query: "bitcoin crypto cryptomonnaie", transaction_type: "debit"

   ✅ RÈGLE CORRIGÉE : "categories" pour catégories de BASE, "query" pour termes non mappés + transaction_type OBLIGATOIRE

5. OPERATION_TYPE (SEULEMENT 6 VALEURS AUTORISÉES) :
   - "paiements par carte" → operation_type: "card"
   - "retraits espèces" → operation_type: "withdrawal"
   - "cartes à débit différé" → operation_type: "deferred_debit_card"
   - "prélèvements automatiques" → operation_type: "direct_debit"
   - "virements" → operation_type: "transfer"
   - "opérations non identifiées" → operation_type: "unknown"
   - "abonnements récurrents" → operation_type: "direct_debit"
   - "paiements contactless" → operation_type: "card"
   - "virements SEPA" → operation_type: "transfer"
   - "chèques" → operation_type: "unknown"
   - RÈGLE : NE PAS INVENTER - utiliser seulement: card, withdrawal, deferred_debit_card, unknown, direct_debit, transfer

=== RÈGLES IMPORTANTES ===

• LOGIQUE PRIORITÉE CORRIGÉE :
  1. MARCHAND spécifique mentionné → merchant: "Nom"
  2. CATÉGORIE de BASE disponible → categories: ["Nom Base"]
  3. TERME non mappé → query: "mots clés"

  EXEMPLES CORRIGÉS:
  - "Mes achats Tesla" → merchant: "Tesla" (marchand spécifique)
  - "Mes achats alimentaires" → categories: ["Supermarkets / Groceries", "Restaurants"] (catégories de base)
  - "Mes frais de transport" → categories: ["Public Transportation", "Taxi/Uber"] (catégories de base)
  - "Mes dépenses santé" → categories: ["Doctor Visits", "Dentist", "Pharmacy"] (catégories de base)
  - "Mes dépenses spatiales" → query: "spatial espace" (pas de catégorie en base)

✅ RAPPEL CORRIGÉ : "categories" AUTORISÉ pour les 57 catégories de BASE uniquement

✅ CAS CORRIGÉS AVEC VRAIES CATÉGORIES :
- "Mes frais de transport" → categories: ["Public Transportation", "Taxi/Uber", "Fuel", "Parking"]
- "Mes dépenses santé" → categories: ["Doctor Visits", "Dentist", "Pharmacy", "Medical Insurance"]
- "Mes factures d'énergie" → categories: ["Electricity", "Water"]

✅ RÈGLE FINALE : categories AUTORISÉ pour les 57 catégories officielles de PostgreSQL

• ACHATS GÉNÉRIQUES (INTERDICTION categories) :
  - "Mes achats" seul → transaction_type: "debit" SEULEMENT (pas de categories, pas de query)
  - "Mes achats" + spécificité → extraire avec 'query', JAMAIS 'categories'
  - INTERDIT: categories: [...]
  - OBLIGATOIRE: query: "mots clés synonymes"

• NORMALISATION AUTOMATIQUE :
  - Corriger les fautes de frappe des marchands
  - Convertir "2024-05" → "mai"
  - Standardiser les montants en euros

{categories_context}

=== FORMAT DE RÉPONSE ===
OBLIGATOIRE : Utiliser JSON OUTPUT uniquement. Format strict :

{{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {{
            "name": "amount",
            "value": 100,
            "confidence": 0.95,
            "span": [15, 25],
            "entity_type": "amount"
        }},
        {{
            "name": "operator",
            "value": "gt",
            "confidence": 0.90,
            "span": [10, 17],
            "entity_type": "operator"
        }},
        {{
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }}
    ],
    "reasoning": "Recherche de dépenses supérieures à 100 euros"
}}

🚨 OBLIGATOIRE: transaction_type TOUJOURS présent dans entities

=== RÈGLES STRICTES ===
- TOUJOURS répondre en JSON valide
- Confidence entre 0.0 et 1.0
- Maximum 10 entités les plus pertinentes
- Si incertain → intent_group: "CONVERSATIONAL"
- Être intelligent et autonome, pas de regex interne
- Comprendre le contexte naturel français
- Corriger automatiquement les erreurs utilisateur"""
    
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

        # ENRICHISSEMENT: Ajouter logique achats/catégories même après succès LLM
        entities = self._enrich_entities_with_purchase_logic(entities, request.user_message)

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
        elif any(word in message_lower for word in ["transaction", "achat", "achats", "depense", "dépense", "dépenses", "paiement", "euro", "euros", "montant"]):
            intent_group = "transaction_search"
            # Détecter si c'est une requête avec montant
            if any(op in message_lower for op in ["plus de", "moins de", "supérieur", "inférieur", "entre", "€", "euros", "euro"]):
                intent_subtype = "by_amount"
                confidence = 0.7
            else:
                intent_subtype = "simple"
                confidence = 0.5
        elif any(merchant in message_lower for merchant in ["amazon", "carrefour", "leclerc", "mcdo", "mcdonald", "netflix", "restaurant", "uber", "fnac", "sncf"]):
            # Si un marchand connu est mentionné, c'est probablement une recherche de transaction
            intent_group = "transaction_search"
            intent_subtype = "by_merchant"
            confidence = 0.8
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
        """Extraction d'entites basique sans LLM - DÉSACTIVÉE au profit de l'agent LLM intelligent"""

        # PLUS AUCUNE EXTRACTION REGEX - TOUT géré par l'agent LLM
        # L'agent LLM doit être assez intelligent pour comprendre :
        # - "Mes achats alimentaires" → catégories alimentaires
        # - "Plus de 500 euros" → montant avec opérateur
        # - "Tesla" → marchand Tesla
        # - etc.

        return []  # Retourner une liste vide - tout géré par LLM

    def _enrich_entities_with_purchase_logic(self, entities: List[ExtractedEntity], message: str) -> List[ExtractedEntity]:
        """Enrichissement désactivé - tout géré par l'agent LLM intelligent"""

        # PLUS AUCUN ENRICHISSEMENT REGEX - TOUT géré par l'agent LLM
        # L'agent LLM doit être assez intelligent pour comprendre directement :
        # - "Mes achats alimentaires" → catégories alimentaires spécifiques
        # - "Mes achats" → catégories multiples automatiques
        # - Normalisation des dates, etc.

        return entities  # Retourner les entités telles que générées par le LLM

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
        # PRIORITE: Examples cas problématiques temporels en positions 1-2 pour être dans top 5
        self._few_shot_examples = [
            {
                "user": "Mes achats du mois de Mai",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_period",
    "confidence": 0.90,
    "entities": [
        {
            "name": "month",
            "value": "mai",
            "confidence": 0.95,
            "span": [19, 22],
            "entity_type": "temporal"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche d'achats dans un mois spécifique - mai"
}"""
            },
            {
                "user": "Toutes mes dépenses du 5 mars",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_date",
    "confidence": 0.90,
    "entities": [
        {
            "name": "date_specific",
            "value": "5 mars",
            "confidence": 0.95,
            "span": [23, 29],
            "entity_type": "temporal"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [10, 18],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de dépenses pour une date spécifique"
}"""
            },
            {
                "user": "Mes achats en ligne",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.85,
    "entities": [
        {
            "name": "categories",
            "value": ["Online Shopping"],
            "confidence": 0.95,
            "span": [10, 19],
            "entity_type": "categories"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Achats en ligne - catégorie 'Online Shopping' disponible en base"
}"""
            },
            {
                "user": "Mes achats Amazon Prime Video Netflix Disney+ Apple TV",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_merchant",
    "confidence": 0.90,
    "entities": [
        {
            "name": "merchants",
            "value": ["Amazon Prime Video", "Netflix", "Disney+", "Apple TV"],
            "confidence": 0.95,
            "span": [10, 52],
            "entity_type": "merchant_list"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche d'achats chez plusieurs marchands spécifiques - services de streaming"
}"""
            },
            {
                "user": "Mes virements de 500 euros ou plus",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {
            "name": "operation_type",
            "value": "transfer",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "operation_type"
        },
        {
            "name": "amount",
            "value": 500,
            "confidence": 0.95,
            "span": [16, 25],
            "entity_type": "amount"
        },
        {
            "name": "operator",
            "value": "gte",
            "confidence": 0.95,
            "span": [26, 34],
            "entity_type": "operator"
        }
    ],
    "reasoning": "Recherche de virements avec montant supérieur ou égal à 500 euros"
}"""
            },
            {
                "user": "Mes dépenses de plus de 100 euros",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {
            "name": "amount",
            "value": 100,
            "confidence": 0.95,
            "span": [17, 27],
            "entity_type": "amount"
        },
        {
            "name": "operator",
            "value": "gt",
            "confidence": 0.95,
            "span": [13, 16],
            "entity_type": "operator"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de dépenses avec montant supérieur à 100 euros"
}"""
            },
            {
                "user": "Mes achats alimentaires",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.90,
    "entities": [
        {
            "name": "categories",
            "value": ["Supermarkets / Groceries", "Restaurants", "Fast foods", "Coffee shop", "Food - Others"],
            "confidence": 0.95,
            "span": [10, 22],
            "entity_type": "categories"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche d'achats alimentaires - utilisation des catégories spécifiques de base"
}"""
            },
            {
                "user": "Mes transactions de 75 euros ou moins",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.90,
    "entities": [
        {
            "name": "amount",
            "value": 75,
            "confidence": 0.95,
            "span": [17, 19],
            "entity_type": "amount"
        },
        {
            "name": "operator",
            "value": "lte",
            "confidence": 0.95,
            "span": [20, 29],
            "entity_type": "operator"
        },
        {
            "name": "transaction_type",
            "value": "all",
            "confidence": 0.90,
            "span": [4, 16],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de toutes transactions avec montant inférieur ou égal à 75 euros"
}"""
            },
            {
                "user": "Mes transactions entre 25 et 75 euros",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_amount",
    "confidence": 0.95,
    "entities": [
        {
            "name": "amount_min",
            "value": 25,
            "confidence": 0.95,
            "span": [20, 22],
            "entity_type": "amount"
        },
        {
            "name": "amount_max",
            "value": 75,
            "confidence": 0.95,
            "span": [26, 28],
            "entity_type": "amount"
        },
        {
            "name": "transaction_type",
            "value": "all",
            "confidence": 0.90,
            "span": [4, 16],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Recherche de transactions dans une plage de montants - structure amount_min/amount_max"
}"""
            },
            {
                "user": "Mes dépenses Amazone",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_merchant",
    "confidence": 0.85,
    "entities": [
        {
            "name": "merchant",
            "value": "Amazon",
            "confidence": 0.95,
            "span": [13, 20],
            "entity_type": "merchant"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Dépenses chez marchand spécifique - normalisation de faute de frappe 'Amazone' vers 'Amazon'"
}"""
            },
            {
                "user": "Mes dépenses spatiales",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.80,
    "entities": [
        {
            "name": "query",
            "value": "spatial espace astronomie cosmos",
            "confidence": 0.85,
            "span": [13, 22],
            "entity_type": "query"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Dépenses spatiales - aucune catégorie spécifique en base, utilisation de query"
}"""
            },
            {
                "user": "Mes dépenses du 15 septembre",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_date",
    "confidence": 0.95,
    "entities": [
        {
            "name": "date_range",
            "value": "2025-09-15",
            "confidence": 0.95,
            "span": [13, 27],
            "entity_type": "temporal"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Date spécifique avec jour et mois - format YYYY-MM-DD"
}"""
            },
            {
                "user": "Mes transactions du 1er au 15 octobre",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_date",
    "confidence": 0.95,
    "entities": [
        {
            "name": "date_range",
            "value": "2025-10-01_2025-10-15",
            "confidence": 0.95,
            "span": [17, 35],
            "entity_type": "temporal"
        },
        {
            "name": "transaction_type",
            "value": "all",
            "confidence": 0.90,
            "span": [4, 16],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Plage de dates spécifiques - format YYYY-MM-DD_YYYY-MM-DD"
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