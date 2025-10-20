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
    # NEW: Analytics detection
    requires_analytics: bool = False
    analytics_type: Optional[str] = None  # "comparison", "trend", "anomaly", "pivot"
    comparison_periods: List[str] = None  # e.g., ["2025-05", "2025-06"]

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

   📅 MOIS SPÉCIFIQUES (on est en octobre 2025) :
   - "de mai" → date_range: "2025-05" (mai 2025, dans le passé)
   - "en janvier 2025" → date_range: "2025-01"
   - "en décembre" → date_range: "2024-12" (décembre dans le futur → année précédente)
   - "d'octobre" → date_range: "2025-10" (mois actuel)
   - "de septembre" → date_range: "2025-09" (mois passé)
   - "de novembre" → date_range: "2024-11" (mois futur → année précédente)

   📍 DATES SPÉCIFIQUES (on est en octobre 2025) :
   - "du 1er mai" → date_range: "2025-05-01" (dans le passé)
   - "du 5 mars" → date_range: "2025-03-05" (dans le passé)
   - "du 15 septembre" → date_range: "2025-09-15" (dans le passé)
   - "du 20 juin" → date_range: "2025-06-20" (dans le passé)
   - "du 31 décembre" → date_range: "2024-12-31" (mois futur → année précédente)

   📊 PLAGES DE DATES (on est en octobre 2025) :
   - "du 14 au 15 mai" → date_range: "2025-05-14_2025-05-15" (dans le passé)
   - "du 1er au 15 octobre" → date_range: "2025-10-01_2025-10-15" (mois actuel)
   - "du 10 au 20 mars" → date_range: "2025-03-10_2025-03-20" (dans le passé)

   🎯 ANNÉES SPÉCIFIQUES :
   - "de 1995" → date_range: "1995"
   - "en 2030" → date_range: "2030"
   - "d'avril 2024" → date_range: "2024-04"

   ⚠️ RÈGLES CRITIQUES ANNÉE CONTEXTUELLE :
   - TOUJOURS utiliser 'date_range' - JAMAIS month, year, date_specific
   - Format strict: YYYY-MM-DD, YYYY-MM, YYYY
   - Dates françaises : "1er" = "01", "5" = "05"
   - Plages avec underscore : "YYYY-MM-DD_YYYY-MM-DD"
   - 🎯 LOGIQUE ANNÉE (on est en octobre 2025) :
     * Si ANNÉE EXPLICITE dans la question → utiliser cette année
     * Si MOIS FUTUR (novembre, décembre) SANS année → utiliser 2024 (année précédente)
     * Si MOIS PASSÉ OU ACTUEL (janvier-octobre) SANS année → utiliser 2025 (année actuelle)
     * Exemples : "décembre" = 2024-12, "septembre" = 2025-09, "novembre" = 2024-11
   - 🚨 DATES INVALIDES (ex: 32 février, 31 avril) :
     * Corriger automatiquement au dernier jour valide du mois
     * "32 février" → date_range: "2025-02-28" (ou 29 si bissextile)
     * "31 avril" → date_range: "2025-04-30"
     * NE PAS classifier comme CONVERSATIONAL si c'est clairement une requête de transaction

3. MARCHANDS ET COMMERÇANTS :
   - "Mes achats [marchand]" → merchant: "[marchand]", transaction_type: "debit"
   - "Mes transactions [marchand]" → merchant: "[marchand]", transaction_type: "all" (ou ne pas mettre)
   - PLUSIEURS marchands : "Amazon Prime Video Netflix Disney+" → merchants: ["Amazon Prime Video", "Netflix", "Disney+"]
   - Corriger automatiquement les fautes de frappe : "Netflik" → "Netflix", "Amazone" → "Amazon"
   - Détecter les marques connues : "Tesla", "Amazon", "McDonald's", "Uber", "Google"
   - Normaliser : "mcdo" → "McDonald's", "Apple/iTunes" → "Apple/iTunes"
   - ⚠️ RÈGLE: transaction_type dépend du contexte (achats→debit, transactions→all, dépenses→debit, revenus→credit)

4. CATÉGORIES → UTILISER LES CATÉGORIES DYNAMIQUES DE LA BASE :
   🎯 RÈGLE PRINCIPALE : Les catégories disponibles sont listées ci-dessous dans le contexte dynamique

   ⚠️ RÈGLES CRITIQUES :
   - Utiliser "categories" UNIQUEMENT pour les catégories listées dans le contexte ci-dessous
   - Pour les termes NON listés dans les catégories → utiliser "query" avec des mots-clés pertinents
   - TOUJOURS inclure transaction_type (debit, credit, ou all)

   📝 EXEMPLES :
   - Si "électronique" N'EST PAS dans les catégories → query: "électronique électro", transaction_type: "debit"
   - Si "Bitcoin" N'EST PAS dans les catégories → query: "bitcoin crypto cryptomonnaie", transaction_type: "debit"
   - Si "spatial" N'EST PAS dans les catégories → query: "spatial espace astronomie", transaction_type: "debit"

5. TYPES DE TRANSACTION vs TYPES D'OPÉRATION (DISTINCTION CRITIQUE) :

   🚨 RÈGLE FONDAMENTALE 🚨
   - transaction_type: UNIQUEMENT "credit" OU "debit" (sens du flux d'argent)
   - operation_type: type d'opération bancaire (carte, virement, retrait, etc.)

   📊 TRANSACTION_TYPE (2 VALEURS SEULEMENT):
   - "credit": argent qui RENTRE (salaire, virement reçu, remboursement, revenus)
   - "debit": argent qui SORT (achats, paiements, retraits, virements sortants, dépenses)

   🏦 OPERATION_TYPE (4 VALEURS STRICTES - LISTE EXACTE DE LA BASE):
   ⚠️ VALEURS AUTORISÉES UNIQUEMENT (casse importante):
   1. "Carte" (majuscule) - 4,099 transactions
   2. "Prélèvement" (majuscule + accents) - 1,360 transactions
   3. "Virement" (majuscule) - 917 transactions
   4. "Chèque" (majuscule + accent) - 167 transactions

   📋 RÈGLES DE MAPPING (requête utilisateur → operation_type):
   - "paiements par carte" / "achats carte" / "paiements contactless" → operation_type: "Carte"
   - "prélèvements automatiques" / "prélèvement SEPA" / "abonnements" → operation_type: "Prélèvement"
   - "virements" / "virements SEPA" / "transferts bancaires" → operation_type: "Virement"
   - "chèques" / "paiements par chèque" → operation_type: "Chèque"

   🚨 RÈGLES CRITIQUES OPERATION_TYPE:
   - NE JAMAIS inventer de valeurs (ex: "card", "direct_debit", "transfer", "withdrawal", "unknown")
   - TOUJOURS utiliser les valeurs EXACTES de la base: "Carte", "Prélèvement", "Virement", "Chèque"
   - Respecter STRICTEMENT la casse (majuscules + accents)
   - NE PAS confondre operation_type avec transaction_type:
     * operation_type = moyen de paiement (Carte, Prélèvement, Virement, Chèque)
     * transaction_type = sens du flux (debit, credit, all)
   - ⚠️ RETRAITS ESPÈCES: Utiliser la catégorie "Retrait especes" (PAS operation_type)
   - Si l'utilisateur ne précise pas le moyen de paiement → NE PAS extraire operation_type

=== RÈGLES IMPORTANTES ===

• LOGIQUE PRIORITÉE :
  1. MARCHAND spécifique mentionné → merchant: "Nom"
  2. CATÉGORIE listée dans le contexte ci-dessous → categories: ["Nom Catégorie"]
  3. TERME non mappé/non listé → query: "mots clés pertinents"

  EXEMPLES:
  - "Mes achats Tesla" → merchant: "Tesla" (marchand spécifique)
  - "Mes dépenses spatiales" → query: "spatial espace" (terme non listé dans les catégories)
  - "Mes achats Bitcoin" → query: "bitcoin crypto" (terme non listé dans les catégories)

✅ RAPPEL : Utiliser UNIQUEMENT les catégories listées dans le contexte dynamique ci-dessous

• ACHATS GÉNÉRIQUES vs ACHATS SPÉCIFIQUES :
  - "Mes achats" SEUL (sans marchand/catégorie/produit/filtre temporel ou autre) → categories: [toutes catégories d'achats listées dans le contexte], transaction_type: "debit"
  - "Mes achats [catégorie spécifique]" (ex: "achats en ligne", "achats alimentaires") → categories: ["[catégorie spécifique]"], transaction_type: "debit"
  - SI période temporelle présente ("du weekend", "de mai", "d'hier", etc.) → NE JAMAIS extraire categories ! Retourner UNIQUEMENT: transaction_type: "debit", date_range: [période]
  - SI marchand présent → merchant: "[marchand]", transaction_type: "debit" (PAS de categories)
  - SI produit spécifique présent → query: "[produit] [mots-clés]", transaction_type: "debit" (PAS de categories)

  ⚠️ ATTENTION CATÉGORIES SPÉCIFIQUES:
  - "Mes achats en ligne" → categories: ["achats en ligne"] SEULEMENT (PAS toutes les catégories d'achats!)
  - "Mes achats alimentaires" → categories: ["Alimentation"] SEULEMENT
  - "Mes achats Tesla" → merchant: "Tesla" (PAS de categories)

  ⚠️ ATTENTION PRODUITS SPÉCIFIQUES :
  - "Mes achats Bitcoin" → query: "bitcoin crypto" (Bitcoin n'est PAS une catégorie, c'est un produit spécifique)
  - "Mes achats Tesla" → merchant: "Tesla" (Tesla est un marchand connu)
  - "Mes achats iPhone" → query: "iphone apple smartphone" (iPhone n'est pas une catégorie)

• TRANSACTIONS NEUTRES :
  - "Mes transactions [marchand]" → merchant: "[marchand]" (PAS de transaction_type ou transaction_type: "all")
  - "Mes transactions" → transaction_type: "all" (toutes transactions)

• NORMALISATION AUTOMATIQUE :
  - Corriger les fautes de frappe des marchands
  - Convertir "2024-05" → "mai"
  - Standardiser les montants en euros

{categories_context}

=== DÉTECTION D'ANALYSES AVANCÉES ===

🔍 DÉTECTION DE COMPARAISON (requires_analytics: true, analytics_type: "comparison"):
   - Mots-clés: "compare", "comparer", "comparaison", "vs", "versus", "différence", "variation"
   - Périodes multiples: "mai vs juin", "ce mois vs mois dernier", "2024 vs 2025"
   - Formulations: "entre mai et juin", "du mois d'avril au mois de mai"

   EXEMPLES:
   - "Compare mes dépenses de mai à celles de juin" → requires_analytics: true, analytics_type: "comparison", comparison_periods: ["2025-05", "2025-06"]
   - "Différence entre mes achats de janvier et février" → requires_analytics: true, analytics_type: "comparison", comparison_periods: ["2025-01", "2025-02"]
   - "Mes dépenses ce mois vs mois dernier" → requires_analytics: true, analytics_type: "comparison", comparison_periods: ["this_month", "last_month"]

🔍 DÉTECTION DE TENDANCE (requires_analytics: true, analytics_type: "trend"):
   - Mots-clés: "évolution", "progression", "tendance", "trend", "historique sur"
   - Formulations: "évolution de mes dépenses", "tendance de mes achats"

   EXEMPLES:
   - "Évolution de mes dépenses restaurants sur 6 mois" → requires_analytics: true, analytics_type: "trend"
   - "Tendance de mes achats en ligne" → requires_analytics: true, analytics_type: "trend"

🔍 DÉTECTION D'ANOMALIE (requires_analytics: true, analytics_type: "anomaly"):
   - Mots-clés: "inhabituel", "anormal", "suspect", "étrange", "bizarre"
   - Formulations: "transactions inhabituelles", "dépenses anormales"

🔍 SI AUCUN CRITÈRE → requires_analytics: false, analytics_type: null

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
    "reasoning": "Recherche de dépenses supérieures à 100 euros",
    "requires_analytics": false,
    "analytics_type": null,
    "comparison_periods": null
}}

🔍 EXEMPLES AVEC ANALYTICS:

1. Question: "Compare mes dépenses de mai à celles de juin"
{{
    "intent_group": "transaction_search",
    "intent_subtype": "comparison",
    "confidence": 0.95,
    "entities": [...],
    "reasoning": "Comparaison des dépenses entre deux mois",
    "requires_analytics": true,
    "analytics_type": "comparison",
    "comparison_periods": ["2025-05", "2025-06"]
}}

2. Question: "Évolution de mes achats en ligne sur 6 mois"
{{
    "intent_group": "transaction_search",
    "intent_subtype": "trend",
    "confidence": 0.93,
    "entities": [...],
    "reasoning": "Analyse de tendance sur 6 mois",
    "requires_analytics": true,
    "analytics_type": "trend",
    "comparison_periods": null
}}

🚨 OBLIGATOIRE:
- transaction_type TOUJOURS présent dans entities
- requires_analytics TOUJOURS présent (true ou false)
- Si requires_analytics: true → analytics_type OBLIGATOIRE ("comparison", "trend", "anomaly", "pivot")

=== RÈGLES STRICTES ===
- TOUJOURS répondre en JSON valide
- Confidence entre 0.0 et 1.0
- Maximum 10 entités les plus pertinentes
- Si incertain → intent_group: "CONVERSATIONAL"
- Être intelligent et autonome, pas de regex interne

⚠️ RÈGLE D'EXCLUSION MUTUELLE CRITIQUE :
On ne peut PAS avoir SIMULTANÉMENT un "query" ET un filtre sur "categories" ou "merchant".
C'est l'un des trois : SOIT query, SOIT categories, SOIT merchant.
- Si on utilise "query" → NE PAS extraire "categories" ni "merchant"
- Si on utilise "merchant" → NE PAS extraire "query" ni "categories"
- Si on utilise "categories" → NE PAS extraire "query" ni "merchant"
- Les autres filtres (date_range, amount, transaction_type, operation_type) sont compatibles avec les 3

EXEMPLES:
- "Mes achats en Bitcoin" → query: "bitcoin crypto" SEULEMENT (PAS de categories)
- "Mes transactions chez Carrefour" → merchant: "Carrefour" SEULEMENT (PAS de query ni categories)
- "Mes achats alimentaires" → categories: ["Alimentation"] SEULEMENT (PAS de query ni merchant)
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

        # VALIDATION POST-LLM: Appliquer règle d'exclusion mutuelle query/categories/merchant
        entities = self._apply_mutual_exclusion_rule(entities, request.user_message)

        # NOUVEAU: Extraction des champs analytics
        requires_analytics = classification_data.get("requires_analytics", False)
        analytics_type = classification_data.get("analytics_type")
        comparison_periods = classification_data.get("comparison_periods")

        # Log si analytics détecté
        if requires_analytics:
            logger.info(f"🔍 Analytics détecté: type={analytics_type}, periods={comparison_periods}")

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
            fallback_used=False,
            requires_analytics=requires_analytics,
            analytics_type=analytics_type,
            comparison_periods=comparison_periods or []
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
        """
        Enrichit les entités avec la logique métier pour les achats et abonnements

        Règles:
        - "Mes achats" (sans marchand) → catégories d'achats
        - "Mes abonnements" (sans marchand) → catégories d'abonnements
        """
        message_lower = message.lower()

        # Nouvelles définitions des achats et abonnements
        purchase_categories = [
            "Carburant", "Transport", "Loisirs", "Entretien maison",
            "achats en ligne", "Alimentation", "Vêtements"
        ]

        subscription_categories = [
            "streaming", "Téléphones/internet", "Services", "Abonnements"
        ]

        # Détecter si c'est une requête "achats" ou "abonnements"
        is_purchase_query = "achat" in message_lower
        is_subscription_query = "abonnement" in message_lower

        if not is_purchase_query and not is_subscription_query:
            return entities

        # Vérifier si un marchand est déjà mentionné dans les entités
        has_merchant = any(e.name in ["merchant", "merchants"] for e in entities)
        if has_merchant:
            return entities  # Si marchand présent, ne pas ajouter de catégories

        # Vérifier si des catégories sont déjà présentes
        has_categories = any(e.name == "categories" for e in entities)
        if has_categories:
            return entities  # Catégories déjà présentes, ne rien faire

        # Choisir les catégories appropriées
        if is_purchase_query:
            target_categories = purchase_categories
            keyword = "achat"
            logger.info(f"Enrichissement 'achats' générique: ajout de {len(target_categories)} catégories: {target_categories}")
        else:  # is_subscription_query
            target_categories = subscription_categories
            keyword = "abonnement"
            logger.info(f"Enrichissement 'abonnements' générique: ajout de {len(target_categories)} catégories: {target_categories}")

        # Ajouter l'entité categories
        categories_entity = ExtractedEntity(
            name="categories",
            value=target_categories,
            confidence=0.90,
            span=(message_lower.find(keyword), message_lower.find(keyword) + len(keyword)),
            entity_type="category"
        )

        return entities + [categories_entity]

    def _apply_mutual_exclusion_rule(self, entities: List[ExtractedEntity], message: str) -> List[ExtractedEntity]:
        """
        Applique la règle d'exclusion mutuelle: query XOR (categories OR merchant)

        Règle métier: On ne peut pas avoir simultanément un "query" ET un filtre sur "categories" ou "merchant"
        - Si "query" existe → supprimer "categories" et "merchant"/"merchants"
        - Garder tous les autres filtres (date_range, amount, transaction_type, operation_type)

        Args:
            entities: Liste des entités extraites par le LLM
            message: Message utilisateur original (pour logging)

        Returns:
            Liste d'entités nettoyée selon la règle d'exclusion
        """
        # Identifier les entités présentes
        has_query = any(e.name == "query" for e in entities)
        has_categories = any(e.name == "categories" for e in entities)
        has_merchant = any(e.name in ["merchant", "merchants", "merchant_name"] for e in entities)

        # Si pas de query, pas de nettoyage nécessaire
        if not has_query:
            return entities

        # Si query existe avec categories ou merchant → supprimer categories et merchant
        if has_query and (has_categories or has_merchant):
            logger.warning(
                f"Règle d'exclusion mutuelle appliquée pour: '{message}' - "
                f"Query présent, suppression de categories={has_categories} et merchant={has_merchant}"
            )

            # Filtrer les entités pour supprimer categories et merchant
            cleaned_entities = [
                e for e in entities
                if e.name not in ["categories", "merchant", "merchants", "merchant_name"]
            ]

            logger.info(
                f"Entités après nettoyage: {[e.name for e in cleaned_entities]} "
                f"(supprimé: {[e.name for e in entities if e not in cleaned_entities]})"
            )

            return cleaned_entities

        # Pas de conflit, retourner tel quel
        return entities

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
                "user": "Mes achats en Bitcoin",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_query",
    "confidence": 0.90,
    "entities": [
        {
            "name": "query",
            "value": "bitcoin crypto cryptomonnaie",
            "confidence": 0.95,
            "span": [15, 22],
            "entity_type": "query"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Bitcoin n'est pas une catégorie - utiliser query SANS categories"
}"""
            },
            {
                "user": "Mes retraits espèces",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.90,
    "entities": [
        {
            "name": "categories",
            "value": ["Retrait especes"],
            "confidence": 0.95,
            "span": [4, 20],
            "entity_type": "categories"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Retraits espèces - utiliser la catégorie 'Retrait especes' spécifique (pas operation_type)"
}"""
            },
            {
                "user": "Mes achats du weekend",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_period",
    "confidence": 0.90,
    "entities": [
        {
            "name": "date_range",
            "value": "weekend",
            "confidence": 0.95,
            "span": [15, 22],
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
    "reasoning": "Recherche d'achats du weekend - SANS categories car période temporelle présente"
}"""
            },
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
                "user": "Mes factures d'énergie",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.90,
    "entities": [
        {
            "name": "categories",
            "value": ["Électricité/eau"],
            "confidence": 0.95,
            "span": [4, 22],
            "entity_type": "categories"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 12],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Factures d'énergie = catégorie Électricité/eau"
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
            "value": ["achats en ligne"],
            "confidence": 0.95,
            "span": [4, 19],
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
    "reasoning": "Achats en ligne - catégorie 'achats en ligne' disponible en base (groupe Vie quotidienne)"
}"""
            },
            {
                "user": "Mes abonnements",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.90,
    "entities": [
        {
            "name": "categories",
            "value": ["streaming", "Téléphones/internet", "Services", "Abonnements"],
            "confidence": 0.95,
            "span": [4, 15],
            "entity_type": "categories"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 15],
            "entity_type": "transaction_type"
        }
    ],
    "reasoning": "Abonnements génériques - regroupe streaming, téléphonie/internet, services et abonnements divers"
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
            "value": "Virement",
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
    "reasoning": "Recherche de virements avec montant >= 500 euros - operation_type=Virement (valeur française correcte)"
}"""
            },
            {
                "user": "Mes achats en ligne du week-end dernier",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_period",
    "confidence": 0.90,
    "entities": [
        {
            "name": "date_range",
            "value": "last_weekend",
            "confidence": 0.95,
            "span": [19, 39],
            "entity_type": "temporal"
        },
        {
            "name": "transaction_type",
            "value": "debit",
            "confidence": 0.95,
            "span": [4, 10],
            "entity_type": "transaction_type"
        },
        {
            "name": "categories",
            "value": ["achats en ligne"],
            "confidence": 0.95,
            "span": [11, 22],
            "entity_type": "category"
        }
    ],
    "reasoning": "Achats en ligne d'une période spécifique - catégorie 'achats en ligne' UNIQUEMENT (pas toutes les catégories!)"
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
                "user": "Mes achats",
                "assistant": """{
    "intent_group": "transaction_search",
    "intent_subtype": "by_category",
    "confidence": 0.90,
    "entities": [
        {
            "name": "categories",
            "value": ["Carburant", "Transport", "Loisirs", "Entretien maison", "achats en ligne", "Alimentation", "Vêtements"],
            "confidence": 0.90,
            "span": [4, 10],
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
    "reasoning": "Achats génériques - regroupe Carburant, Transport, Loisirs, Entretien maison, achats en ligne, Alimentation et Vêtements"
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
            "value": ["Alimentation"],
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
    "reasoning": "Achats alimentaires - catégorie spécifique Alimentation du groupe Vie quotidienne"
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