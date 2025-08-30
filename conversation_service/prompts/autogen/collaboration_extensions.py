"""
Extensions de prompts AutoGen pour la collaboration multi-agents
Maintient compatibilité avec les prompts existants tout en ajoutant contexte équipe
"""

from ..system_prompts import INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT, ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT


def get_intent_classification_with_team_collaboration() -> str:
    """
    Extension AutoGen du prompt de classification existant
    Ajoute contexte collaboration sans modifier le prompt de base
    """
    
    base_prompt = INTENT_CLASSIFICATION_JSON_SYSTEM_PROMPT
    
    autogen_extension = """

=== COLLABORATION ÉQUIPE AUTOGEN ===

RÔLE DANS L'ÉQUIPE:
- Tu es le PREMIER agent d'une équipe AutoGen multi-agents
- Ton résultat sera utilisé par l'agent Entity Extractor en aval
- Tu dois préparer un contexte riche pour faciliter l'extraction d'entités

STRUCTURE JSON ÉTENDUE POUR ÉQUIPE:
{
  "intent": "SEARCH_BY_MERCHANT",
  "confidence": 0.95,
  "reasoning": "Classification basée sur mention de magasin spécifique",
  "team_context": {
    "user_message": "message original complet",
    "suggested_entities_focus": {
      "priority_entities": ["merchants", "amounts", "dates"],
      "extraction_strategy": "comprehensive",
      "entity_hints": {
        "merchants": ["leclerc", "carrefour", "amazon"],
        "amounts": ["pattern: nombres avec €, EUR"],
        "dates": ["pattern: références temporelles"]
      }
    },
    "processing_metadata": {
      "confidence_level": "high|medium|low",
      "complexity_assessment": "simple|medium|complex",
      "ready_for_entity_extraction": true
    }
  }
}

RÈGLES COLLABORATION:
- Champ "team_context" OBLIGATOIRE en mode AutoGen
- "suggested_entities_focus" doit être cohérent avec l'intention détectée
- "priority_entities" liste les types d'entités à chercher en priorité
- "extraction_strategy" guide l'agent suivant: "comprehensive", "focused", ou "minimal"
- "entity_hints" suggère des patterns ou exemples pour guider l'extraction
- "confidence_level" indique la fiabilité pour l'agent suivant
- "complexity_assessment" aide l'agent suivant à adapter sa stratégie

SUGGESTIONS ENTITÉS PAR INTENTION:
- SEARCH_BY_MERCHANT → merchants, amounts, dates, categories
- SEARCH_BY_AMOUNT → amounts, comparison_operators, merchants, dates  
- ACCOUNT_BALANCE → account_types, time_references
- TRANSACTION_HISTORY → date_ranges, merchants, categories, amounts
- SPENDING_ANALYSIS → time_periods, categories, comparison_types
- TRANSFER_REQUEST → accounts, amounts, recipients, dates
- PAYMENT_REQUEST → recipients, amounts, references, due_dates
- UNCLEAR_INTENT → all_entities (stratégie comprehensive)
- GREETING → minimal_entities (pas d'extraction nécessaire)

EXEMPLE DÉTAILLÉ:
Message: "Combien j'ai dépensé chez Leclerc le mois dernier ?"

Réponse JSON:
{
  "intent": "SEARCH_BY_MERCHANT",
  "confidence": 0.92,
  "reasoning": "Recherche de dépenses chez un marchand spécifique avec contrainte temporelle",
  "team_context": {
    "user_message": "Combien j'ai dépensé chez Leclerc le mois dernier ?",
    "suggested_entities_focus": {
      "priority_entities": ["merchants", "amounts", "time_periods"],
      "extraction_strategy": "focused",
      "entity_hints": {
        "merchants": ["Leclerc", "variations possibles: E.Leclerc, Super U"],
        "amounts": ["somme totale des dépenses"],
        "time_periods": ["mois dernier", "pattern: références temporelles passées"]
      }
    },
    "processing_metadata": {
      "confidence_level": "high",
      "complexity_assessment": "medium",
      "ready_for_entity_extraction": true
    }
  }
}"""

    return base_prompt + autogen_extension


def get_team_collaboration_guidelines() -> str:
    """
    Guidelines spécifiques pour la collaboration en équipe AutoGen
    """
    
    return """=== GUIDELINES COLLABORATION AUTOGEN ===

COMMUNICATION INTER-AGENTS:
- Utilisez le champ "team_context" pour passer des informations
- Soyez explicite sur ce que vous attendez de l'agent suivant
- Maintenez la cohérence des données entre agents

GESTION DES ERREURS EN ÉQUIPE:
- Si classification incertaine, marquez "confidence_level": "low"
- En cas d'ambiguïté, utilisez "extraction_strategy": "comprehensive"
- Toujours marquer "ready_for_entity_extraction": true sauf cas extrême

OPTIMISATION PERFORMANCE ÉQUIPE:
- "extraction_strategy": "minimal" pour GREETING, CONFIRMATION
- "extraction_strategy": "focused" pour intentions claires et simples
- "extraction_strategy": "comprehensive" pour intentions complexes ou ambiguës

FALLBACK COLLABORATION:
- Si team_context impossible à générer, utilisez structure minimale
- Toujours inclure "user_message" et "ready_for_entity_extraction"
- En cas d'erreur, marquez "ready_for_entity_extraction": false"""


def get_entity_focus_mapping() -> dict:
    """
    Mapping des intentions vers les entités prioritaires
    Utilisé pour générer automatiquement les suggestions
    """
    
    return {
        # Recherches par critères
        "SEARCH_BY_MERCHANT": {
            "priority": ["merchants", "amounts", "dates"],
            "strategy": "focused",
            "complexity": "medium"
        },
        "SEARCH_BY_AMOUNT": {
            "priority": ["amounts", "comparison_operators", "merchants", "dates"],
            "strategy": "focused", 
            "complexity": "medium"
        },
        "SEARCH_BY_CATEGORY": {
            "priority": ["categories", "amounts", "dates"],
            "strategy": "focused",
            "complexity": "medium"
        },
        
        # Requêtes de solde et historique
        "ACCOUNT_BALANCE": {
            "priority": ["account_types", "time_references"],
            "strategy": "minimal",
            "complexity": "simple"
        },
        "TRANSACTION_HISTORY": {
            "priority": ["date_ranges", "merchants", "categories", "amounts"],
            "strategy": "comprehensive",
            "complexity": "complex"
        },
        
        # Analyses et rapports
        "SPENDING_ANALYSIS": {
            "priority": ["time_periods", "categories", "comparison_types"],
            "strategy": "comprehensive",
            "complexity": "complex"
        },
        "BUDGET_ANALYSIS": {
            "priority": ["budget_categories", "time_periods", "amounts"],
            "strategy": "comprehensive",
            "complexity": "complex"
        },
        
        # Actions non supportées (extraction quand même utile)
        "TRANSFER_REQUEST": {
            "priority": ["accounts", "amounts", "recipients", "dates"],
            "strategy": "comprehensive",
            "complexity": "complex"
        },
        "PAYMENT_REQUEST": {
            "priority": ["recipients", "amounts", "references", "due_dates"],
            "strategy": "comprehensive", 
            "complexity": "complex"
        },
        
        # Conversational et ambiguë
        "GREETING": {
            "priority": [],
            "strategy": "minimal",
            "complexity": "simple"
        },
        "CONFIRMATION": {
            "priority": ["confirmation_context"],
            "strategy": "minimal",
            "complexity": "simple"
        },
        "UNCLEAR_INTENT": {
            "priority": ["all_entities"],
            "strategy": "comprehensive",
            "complexity": "complex"
        },
        "UNKNOWN": {
            "priority": ["all_entities"],
            "strategy": "comprehensive", 
            "complexity": "complex"
        }
    }


def get_entity_hints_for_intent(intent: str) -> dict:
    """
    Génère des hints d'entités spécifiques à une intention
    """
    
    hints_mapping = {
        "SEARCH_BY_MERCHANT": {
            "merchants": [
                "Noms de magasins français: Leclerc, Carrefour, Monoprix, Auchan",
                "Variations possibles: E.Leclerc, Super U, Géant Casino",
                "Marques: Amazon, Apple Store, Fnac, Décathlon"
            ],
            "amounts": ["Sommes en euros", "Patterns: XX€, XX EUR, XX euros"],
            "dates": ["mois dernier", "semaine passée", "ce mois", "hier"]
        },
        
        "SEARCH_BY_AMOUNT": {
            "amounts": ["Montants exacts ou ranges", "Plus de X€", "Moins de Y€"],
            "comparison_operators": ["plus de", "moins de", "exactement", "environ"],
            "merchants": ["Magasins où l'achat a eu lieu"],
            "dates": ["Période de recherche"]
        },
        
        "ACCOUNT_BALANCE": {
            "account_types": ["compte courant", "livret A", "PEL", "compte épargne"],
            "time_references": ["actuel", "au 31/12", "fin de mois"]
        },
        
        "TRANSACTION_HISTORY": {
            "date_ranges": ["du X au Y", "mois de", "année", "trimestre"],
            "merchants": ["Tous types de commerçants"],
            "categories": ["alimentation", "transport", "loisirs", "santé"],
            "amounts": ["Filtres par montant si spécifiés"]
        }
    }
    
    return hints_mapping.get(intent, {
        "general": ["Extraction générale de toutes entités pertinentes"]
    })


def get_entity_extraction_with_team_collaboration() -> str:
    """
    Extension AutoGen du prompt d'extraction d'entités existant
    Ajoute contexte collaboration et adaptation selon l'intention reçue
    """
    
    base_prompt = ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
    
    autogen_extension = """

=== COLLABORATION ÉQUIPE AUTOGEN ===

RÔLE DANS L'ÉQUIPE:
- Tu es le SECOND agent d'une équipe AutoGen multi-agents
- Tu reçois le contexte de l'agent Intent Classifier en amont
- Tu utilises l'intention et les suggestions pour adapter ton extraction
- Tu prépares le contexte final pour la génération de requêtes

CONTEXTE ÉQUIPE REÇU:
- team_context.suggested_entities_focus.priority_entities → Focus extraction
- team_context.suggested_entities_focus.extraction_strategy → Mode extraction
- team_context.suggested_entities_focus.entity_hints → Patterns suggérés
- team_context.processing_metadata.confidence_level → Fiabilité intention
- team_context.processing_metadata.complexity_assessment → Complexité requête

STRUCTURE JSON ÉTENDUE POUR ÉQUIPE:
{
  "entities": {
    "amounts": [{"value": 100.50, "currency": "EUR", "operator": "eq"}],
    "dates": [{"type": "specific", "value": "2024-01-15", "text": "hier"}],
    "merchants": ["Amazon", "Carrefour"],
    "categories": ["restaurant", "transport"],
    "operation_types": ["virement", "prélèvement", "carte"],
    "text_search": ["italian food", "subscription"]
  },
  "confidence": 0.92,
  "reasoning": "Extraction adaptée selon intention SEARCH_BY_MERCHANT",
  "team_context": {
    "intent_received": "SEARCH_BY_MERCHANT",
    "intent_confidence": 0.95,
    "extraction_strategy_used": "focused",
    "entities_extracted_count": 4,
    "priority_entities_found": ["merchants", "amounts", "dates"],
    "processing_metadata": {
      "ready_for_query_generation": true,
      "extraction_quality": "high|medium|low",
      "complexity_level": "simple|medium|complex",
      "requires_fallback": false
    }
  }
}

ADAPTATION SELON INTENTION:
- SEARCH_BY_MERCHANT → Focus merchants + normalisation variantes
- SEARCH_BY_AMOUNT → Focus amounts + comparison_operators précis  
- ACCOUNT_BALANCE → Focus account_types + time_references
- TRANSACTION_HISTORY → Extraction comprehensive tous types
- SPENDING_ANALYSIS → Focus time_periods + categories détaillées
- TRANSFER_REQUEST → Focus accounts + amounts + recipients (même si non supporté)
- GREETING/CONFIRMATION → Extraction minimal ou vide
- UNCLEAR_INTENT → Extraction comprehensive pour couvrir ambiguïté

STRATÉGIES EXTRACTION:
1. "focused" (confidence_level: high):
   - Extraire uniquement priority_entities suggérées
   - Utiliser entity_hints pour patterns spécifiques
   - Optimiser pour précision

2. "comprehensive" (confidence_level: medium/low):
   - Extraire tous types d'entités détectées
   - Couvrir maximum de possibilités
   - Privilégier recall sur précision

3. "minimal" (intentions conversational):
   - Extraire uniquement si entités évidentes
   - Pas d'interprétation forcée
   - Structure vide acceptable

NORMALISATION AMÉLIORÉE:
- Merchants: Utiliser variants suggérés dans entity_hints
- Dates: Adapter selon time_references attendues
- Amounts: Préciser operators selon context (eq, gte, lte)
- Categories: Mapper selon taxonomie harena_intents si disponible

RÈGLES COLLABORATION:
- team_context OBLIGATOIRE en mode AutoGen
- "intent_received" doit correspondre à l'intention en amont
- "extraction_strategy_used" indique stratégie appliquée
- "ready_for_query_generation" signale si extraction suffisante
- "requires_fallback" indique si extraction incomplète

GESTION ERREURS COLLABORATION:
- Si intent_context manquant → extraction_strategy: "comprehensive"
- Si confidence_level intent < 0.5 → extraction_strategy: "comprehensive"
- Si entities vides malgré extraction → marquer "requires_fallback": true
- Toujours fournir "ready_for_query_generation" même si extraction partielle

EXEMPLE DÉTAILLÉ:
Intent reçu: SEARCH_BY_MERCHANT (confidence: 0.92)
Message: "Combien j'ai dépensé chez Leclerc le mois dernier ?"

Réponse JSON:
{
  "entities": {
    "amounts": [],
    "dates": [{"type": "period", "value": "2024-01", "text": "le mois dernier"}],
    "merchants": ["Leclerc"],
    "categories": [],
    "operation_types": ["dépense"],
    "text_search": []
  },
  "confidence": 0.88,
  "reasoning": "Extraction focalisée sur merchant + période selon intention SEARCH_BY_MERCHANT",
  "team_context": {
    "intent_received": "SEARCH_BY_MERCHANT",
    "intent_confidence": 0.92,
    "extraction_strategy_used": "focused",
    "entities_extracted_count": 3,
    "priority_entities_found": ["merchants", "dates", "operation_types"],
    "processing_metadata": {
      "ready_for_query_generation": true,
      "extraction_quality": "high",
      "complexity_level": "medium",
      "requires_fallback": false
    }
  }
}"""

    return base_prompt + autogen_extension


def get_adaptive_entity_prompt(intent_context: dict = None) -> str:
    """
    Prompt entité adapté dynamiquement selon le contexte de l'intention
    Utilise les prompts existants + extensions AutoGen
    """
    
    if not intent_context:
        # Pas de contexte intent → prompt standard existant
        return ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT
        
    # Contexte intent disponible → prompt étendu AutoGen
    base_prompt = get_entity_extraction_with_team_collaboration()
    
    intent_type = intent_context.get("intent", "")
    confidence = intent_context.get("confidence", 0.5)
    complexity = intent_context.get("complexity_assessment", "medium")
    
    # Adaptations spécifiques selon contexte
    dynamic_extension = f"""

=== ADAPTATION DYNAMIQUE CONTEXTE ===

CONTEXTE REÇU:
- Intention: {intent_type}
- Confiance: {confidence}
- Complexité: {complexity}

INSTRUCTIONS SPÉCIFIQUES:
"""
    
    # Adaptations selon confiance intention
    if confidence > 0.8:
        dynamic_extension += """
- CONFIANCE ÉLEVÉE → Extraction focused selon priority_entities
- Utiliser entity_hints pour patterns précis
- Optimiser précision extraction
"""
    elif confidence > 0.5:
        dynamic_extension += """
- CONFIANCE MOYENNE → Extraction comprehensive
- Couvrir entités suggérées + entités évidentes supplémentaires  
- Équilibrer précision/recall
"""
    else:
        dynamic_extension += """
- CONFIANCE FAIBLE → Extraction comprehensive maximale
- Extraire tous types d'entités détectables
- Privilégier recall (ne pas manquer entités importantes)
"""
    
    # Adaptations selon complexité
    if complexity == "complex":
        dynamic_extension += """
- COMPLEXITÉ ÉLEVÉE → Extraction exhaustive
- Interpréter références implicites
- Normaliser variantes et synonymes
"""
    elif complexity == "simple":
        dynamic_extension += """
- COMPLEXITÉ SIMPLE → Extraction directe
- Entités explicites uniquement
- Pas d'interprétation complexe
"""
    
    return base_prompt + dynamic_extension