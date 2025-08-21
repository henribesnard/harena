"""
Complete intent taxonomy for Harena Conversation Service.

This module defines the comprehensive taxonomy of intentions for Harena's
consultation-focused banking assistant. It includes classification rules,
unsupported action detection, and intelligent redirection strategies.

Key Features:
- 6 primary categories aligned with Harena's consultation scope
- Automatic detection of unsupported actions with redirection
- Keyword-based classification fallback system
- Response templates for consistent user experience
- Confidence scoring guidelines for each intent type

Author: Harena Conversation Team  
Created: 2025-01-31
Version: 1.0.0 - Harena Consultation Scope
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from ..models.enums import IntentType

__all__ = [
    "IntentTaxonomy", "IntentClassificationMatrix", "UnsupportedActionDetector",
    "HarenaResponseTemplates", "ConfidenceThresholds"
]

# ================================
# CONFIDENCE THRESHOLDS
# ================================

class ConfidenceThresholds:
    """Confidence thresholds for different intent categories."""
    
    # Supported intents (Harena can handle)
    CONSULTATION_MIN = 0.5
    ANALYSIS_MIN = 0.5
    INFORMATION_MIN = 0.5
    CONVERSATIONAL_MIN = 0.7
    
    # Unsupported actions (require high confidence for redirection)
    UNSUPPORTED_ACTION_MIN = 0.8
    
    # Out of scope / unknown
    OUT_OF_SCOPE_MAX = 0.6
    UNKNOWN_MAX = 0.5
    AMBIGUOUS_MAX = 0.5
    
    # Quality gates
    REJECT_THRESHOLD = 0.3  # Below this, return UNKNOWN
    HIGH_CONFIDENCE = 0.9   # Above this, very confident classification

# ================================
# INTENT CLASSIFICATION MATRIX
# ================================

@dataclass
class IntentClassificationRule:
    """Rule for keyword-based intent classification."""
    keywords: List[str]
    intent: IntentType
    confidence_boost: float = 0.1
    required_all: bool = False  # True if all keywords required
    context_hints: Optional[List[str]] = None

class IntentClassificationMatrix:
    """
    Keyword-based classification matrix for fallback and validation.
    
    Provides rule-based classification when LLM is unavailable or
    for validation of LLM results. Organized by Harena scope categories.
    """
    
    # === CONSULTATION KEYWORDS ===
    CONSULTATION_RULES = [
        IntentClassificationRule(
            keywords=["solde", "balance", "combien", "argent", "compte"],
            intent=IntentType.BALANCE_INQUIRY,
            confidence_boost=0.2,
            context_hints=["actuel", "disponible", "sur", "mon"]
        ),
        IntentClassificationRule(
            keywords=["transactions", "opérations", "mouvements", "historique"],
            intent=IntentType.TRANSACTION_SEARCH,
            confidence_boost=0.15,
            context_hints=["dernières", "récentes", "voir", "afficher"]
        ),
        IntentClassificationRule(
            keywords=["relevé", "statement", "récapitulatif", "synthèse"],
            intent=IntentType.STATEMENT_REQUEST,
            confidence_boost=0.2,
            context_hints=["mois", "période", "télécharger"]
        ),
        IntentClassificationRule(
            keywords=["comptes", "accounts", "vue", "ensemble", "overview"],
            intent=IntentType.ACCOUNT_OVERVIEW,
            confidence_boost=0.15,
            context_hints=["tous", "mes", "aperçu", "global"]
        )
    ]
    
    # === ANALYSIS KEYWORDS ===
    ANALYSIS_RULES = [
        IntentClassificationRule(
            keywords=["dépenses", "spending", "dépensé", "coûté"],
            intent=IntentType.SPENDING_ANALYSIS,
            confidence_boost=0.2,
            context_hints=["mois", "année", "période", "total"]
        ),
        IntentClassificationRule(
            keywords=["restaurant", "alimentation", "transport", "shopping", "santé"],
            intent=IntentType.CATEGORY_ANALYSIS,
            confidence_boost=0.25,
            context_hints=["catégorie", "en", "pour", "dépenses"]
        ),
        IntentClassificationRule(
            keywords=["amazon", "carrefour", "fnac", "sncf", "uber", "chez"],
            intent=IntentType.MERCHANT_ANALYSIS,
            confidence_boost=0.2,
            context_hints=["chez", "sur", "avec", "achats"]
        ),
        IntentClassificationRule(
            keywords=["évolution", "tendance", "progression", "comparaison"],
            intent=IntentType.TEMPORAL_ANALYSIS,
            confidence_boost=0.2,
            context_hints=["vs", "versus", "par rapport", "depuis"]
        ),
        IntentClassificationRule(
            keywords=["budget", "budgétaire", "prévisionnel", "objectif"],
            intent=IntentType.BUDGET_ANALYSIS,
            confidence_boost=0.2,
            context_hints=["vs", "réalisé", "dépassé", "respecté"]
        ),
        IntentClassificationRule(
            keywords=["revenus", "salaire", "income", "gains"],
            intent=IntentType.INCOME_ANALYSIS,
            confidence_boost=0.2,
            context_hints=["mois", "année", "net", "brut"]
        )
    ]
    
    # === INFORMATION & SUPPORT KEYWORDS ===
    INFORMATION_RULES = [
        IntentClassificationRule(
            keywords=["comment", "expliquer", "explication", "help", "aide"],
            intent=IntentType.HELP_REQUEST,
            confidence_boost=0.15,
            context_hints=["fonctionne", "marche", "comprendre", "utiliser"]
        ),
        IntentClassificationRule(
            keywords=["frais", "commission", "fees", "coût"],
            intent=IntentType.FEE_INQUIRY,
            confidence_boost=0.2,
            context_hints=["carte", "compte", "virement", "combien"]
        ),
        IntentClassificationRule(
            keywords=["livret", "épargne", "pea", "assurance", "produit"],
            intent=IntentType.PRODUCT_INFORMATION,
            confidence_boost=0.2,
            context_hints=["qu'est-ce", "c'est quoi", "fonctionne", "taux"]
        ),
        IntentClassificationRule(
            keywords=["informations", "détails", "renseignements"],
            intent=IntentType.ACCOUNT_INFORMATION,
            confidence_boost=0.1,
            context_hints=["compte", "carte", "produit", "service"]
        )
    ]
    
    # === CONVERSATIONAL KEYWORDS ===
    CONVERSATIONAL_RULES = [
        IntentClassificationRule(
            keywords=["bonjour", "hello", "salut", "bonsoir", "hey"],
            intent=IntentType.GREETING,
            confidence_boost=0.3,
            required_all=False
        ),
        IntentClassificationRule(
            keywords=["au revoir", "goodbye", "bye", "à bientôt", "ciao"],
            intent=IntentType.GOODBYE,
            confidence_boost=0.3,
            required_all=False
        ),
        IntentClassificationRule(
            keywords=["merci", "thank", "thanks", "remercie"],
            intent=IntentType.THANKS,
            confidence_boost=0.3,
            context_hints=["beaucoup", "bien", "vous"]
        ),
        IntentClassificationRule(
            keywords=["répéter", "repeat", "comprends", "pardon", "quoi"],
            intent=IntentType.CLARIFICATION_REQUEST,
            confidence_boost=0.2,
            context_hints=["pas", "n'ai", "pouvez", "encore"]
        ),
        IntentClassificationRule(
            keywords=["comment", "allez", "vous", "ça va", "how are"],
            intent=IntentType.POLITENESS,
            confidence_boost=0.2,
            required_all=False
        )
    ]
    
    # === UNSUPPORTED ACTION KEYWORDS ===
    UNSUPPORTED_ACTION_RULES = [
        IntentClassificationRule(
            keywords=["virement", "virer", "transférer", "transfer", "envoyer"],
            intent=IntentType.TRANSFER_REQUEST,
            confidence_boost=0.3,
            context_hints=["faire", "effectuer", "vers", "euro", "€"]
        ),
        IntentClassificationRule(
            keywords=["payer", "payment", "régler", "facture", "bill"],
            intent=IntentType.PAYMENT_REQUEST,
            confidence_boost=0.3,
            context_hints=["facture", "edf", "orange", "loyer"]
        ),
        IntentClassificationRule(
            keywords=["bloquer", "débloquer", "activer", "désactiver", "opposition"],
            intent=IntentType.CARD_OPERATIONS,
            confidence_boost=0.3,
            context_hints=["carte", "card", "pin", "code"]
        ),
        IntentClassificationRule(
            keywords=["crédit", "prêt", "loan", "emprunt", "financement"],
            intent=IntentType.LOAN_REQUEST,
            confidence_boost=0.3,
            context_hints=["demande", "simulation", "taux", "immobilier"]
        ),
        IntentClassificationRule(
            keywords=["modifier", "changer", "mettre à jour", "update"],
            intent=IntentType.ACCOUNT_MODIFICATION,
            confidence_boost=0.3,
            context_hints=["adresse", "téléphone", "email", "coordonnées"]
        ),
        IntentClassificationRule(
            keywords=["acheter", "vendre", "investir", "bourse", "actions"],
            intent=IntentType.INVESTMENT_OPERATIONS,
            confidence_boost=0.3,
            context_hints=["actions", "etf", "obligation", "placement"]
        )
    ]
    
    @classmethod
    def get_all_rules(cls) -> List[IntentClassificationRule]:
        """Get all classification rules."""
        return (
            cls.CONSULTATION_RULES + 
            cls.ANALYSIS_RULES + 
            cls.INFORMATION_RULES + 
            cls.CONVERSATIONAL_RULES + 
            cls.UNSUPPORTED_ACTION_RULES
        )
    
    @classmethod
    def classify_by_keywords(cls, user_message: str) -> List[Tuple[IntentType, float]]:
        """
        Classify intent using keyword rules.
        
        Returns list of (intent, confidence) tuples sorted by confidence.
        """
        message_lower = user_message.lower()
        candidates = []
        
        for rule in cls.get_all_rules():
            score = 0.0
            keyword_matches = 0
            
            # Check main keywords
            for keyword in rule.keywords:
                if keyword in message_lower:
                    keyword_matches += 1
                    score += rule.confidence_boost
            
            # Apply required_all constraint
            if rule.required_all and keyword_matches < len(rule.keywords):
                continue
            
            # Skip if no keywords matched
            if keyword_matches == 0:
                continue
            
            # Check context hints for additional confidence
            if rule.context_hints:
                context_matches = sum(1 for hint in rule.context_hints if hint in message_lower)
                context_boost = min(context_matches * 0.05, 0.2)  # Max 0.2 boost
                score += context_boost
            
            # Normalize score by keyword count for fairness
            normalized_score = score * (keyword_matches / len(rule.keywords))
            
            # Cap score at reasonable maximum
            final_score = min(normalized_score, 0.8)
            
            if final_score > 0.1:  # Minimum threshold
                candidates.append((rule.intent, final_score))
        
        # Sort by confidence descending
        return sorted(candidates, key=lambda x: x[1], reverse=True)

# ================================
# UNSUPPORTED ACTION DETECTOR
# ================================

class UnsupportedActionDetector:
    """
    Specialized detector for actions not supported by Harena.
    
    Provides high-precision detection of transactional requests
    that need to be redirected to main banking services.
    """
    
    # Action verb patterns with confidence weights
    ACTION_PATTERNS = {
        # Transfer/payment verbs (weight: 0.4)
        r'\b(faire|effectuer|réaliser)\s+(?:un\s+)?(virement|transfer|paiement)': 0.4,
        r'\b(virer|transférer|envoyer)\s+(?:\d+|\w+)': 0.4,
        r'\b(payer|régler)\s+(?:la\s+|ma\s+|une\s+)?facture': 0.4,
        
        # Card operations (weight: 0.3)
        r'\b(bloquer|débloquer|activer|désactiver)\s+(?:ma\s+)?carte': 0.3,
        r'\bopposition\s+(?:sur\s+)?carte': 0.3,
        r'\bchanger\s+(?:le\s+|mon\s+)?code\s+pin': 0.3,
        
        # Account modifications (weight: 0.3)
        r'\b(modifier|changer|mettre\s+à\s+jour)\s+(?:mes?\s+)?(?:coordonnées|adresse|téléphone)': 0.3,
        r'\bnouvelle?\s+adresse': 0.2,
        
        # Investment operations (weight: 0.3)
        r'\b(acheter|vendre|investir)\s+(?:des?\s+)?(?:actions|etf|obligations)': 0.3,
        r'\bplacer\s+(?:\d+|\w+)\s+euros?': 0.3,
        
        # Loan requests (weight: 0.3)
        r'\b(demande|simulation)\s+(?:de\s+)?(?:crédit|prêt)': 0.3,
        r'\bemprunter\s+(?:\d+|\w+)': 0.3,
        
        # Amount + action context (weight: 0.2)
        r'\b\d+\s*(?:€|euros?)\s+(?:vers|pour|à)': 0.2,
        r'\b(?:vers|sur)\s+(?:le\s+)?compte': 0.2
    }
    
    # Context patterns that increase action confidence
    CONTEXT_BOOSTERS = {
        r'\burgent\b': 0.1,
        r'\bimmédiatement\b': 0.1,
        r'\bmaintenant\b': 0.1,
        r'\baujourd\'hui\b': 0.05,
        r'\brapidement\b': 0.05
    }
    
    # Amount detection patterns
    AMOUNT_PATTERNS = [
        r'\b\d+(?:[.,]\d+)?\s*(?:€|euros?)\b',
        r'\b\d+(?:[.,]\d+)?\s*(?:\$|dollars?)\b',
        r'\b(?:€|euros?)\s*\d+(?:[.,]\d+)?\b'
    ]
    
    @classmethod
    def detect_unsupported_action(cls, user_message: str) -> Tuple[bool, float, List[str]]:
        """
        Detect if message contains unsupported action request.
        
        Returns:
            - is_action: True if action detected
            - confidence: Confidence score (0.0-1.0)
            - detected_patterns: List of matched patterns for explanation
        """
        message_lower = user_message.lower()
        total_score = 0.0
        detected_patterns = []
        
        # Check action patterns
        for pattern, weight in cls.ACTION_PATTERNS.items():
            matches = re.findall(pattern, message_lower)
            if matches:
                total_score += weight
                detected_patterns.append(f"Action pattern: {pattern}")
        
        # Check context boosters
        for pattern, boost in cls.CONTEXT_BOOSTERS.items():
            if re.search(pattern, message_lower):
                total_score += boost
                detected_patterns.append(f"Context booster: {pattern}")
        
        # Amount detection adds confidence to actions
        amount_found = any(re.search(pattern, message_lower) for pattern in cls.AMOUNT_PATTERNS)
        if amount_found and total_score > 0:
            total_score += 0.1
            detected_patterns.append("Amount specified with action")
        
        # Normalize confidence (cap at 0.95)
        confidence = min(total_score, 0.95)
        is_action = confidence >= ConfidenceThresholds.UNSUPPORTED_ACTION_MIN
        
        return is_action, confidence, detected_patterns
    
    @classmethod
    def categorize_action_type(cls, user_message: str) -> Optional[IntentType]:
        """Categorize the specific type of unsupported action."""
        message_lower = user_message.lower()
        
        # Transfer indicators
        transfer_keywords = ['virement', 'virer', 'transférer', 'transfer', 'envoyer']
        if any(keyword in message_lower for keyword in transfer_keywords):
            return IntentType.TRANSFER_REQUEST
        
        # Payment indicators
        payment_keywords = ['payer', 'régler', 'facture', 'payment']
        if any(keyword in message_lower for keyword in payment_keywords):
            return IntentType.PAYMENT_REQUEST
        
        # Card operation indicators
        card_keywords = ['bloquer', 'débloquer', 'carte', 'opposition', 'pin']
        if any(keyword in message_lower for keyword in card_keywords):
            return IntentType.CARD_OPERATIONS
        
        # Account modification indicators
        modification_keywords = ['modifier', 'changer', 'adresse', 'téléphone', 'coordonnées']
        if any(keyword in message_lower for keyword in modification_keywords):
            return IntentType.ACCOUNT_MODIFICATION
        
        # Loan indicators
        loan_keywords = ['crédit', 'prêt', 'emprunt', 'financement']
        if any(keyword in message_lower for keyword in loan_keywords):
            return IntentType.LOAN_REQUEST
        
        # Investment indicators
        investment_keywords = ['acheter', 'vendre', 'investir', 'actions', 'bourse']
        if any(keyword in message_lower for keyword in investment_keywords):
            return IntentType.INVESTMENT_OPERATIONS
        
        return IntentType.UNSUPPORTED_ACTION

# ================================
# HARENA RESPONSE TEMPLATES
# ================================

class HarenaResponseTemplates:
    """
    Response templates for consistent handling of unsupported actions.
    
    Provides professional, helpful responses that explain Harena's
    limitations and guide users to appropriate alternatives.
    """
    
    REDIRECTION_TEMPLATES = {
        IntentType.TRANSFER_REQUEST: {
            "template": """
Je ne peux pas effectuer de virements pour vous. Harena est un assistant consultatif qui vous aide à analyser vos finances.

**Pour vos virements :**
• Application mobile de votre banque
• Espace client en ligne sécurisé  
• Contactez votre conseiller bancaire

**Je peux vous aider avec :**
• Consulter l'historique de vos virements
• Analyser vos transferts par période
• Voir le détail de vos comptes

Souhaitez-vous voir vos derniers virements ou analyser vos transferts ?
            """.strip(),
            "suggestions": [
                "Voir mes derniers virements",
                "Analyser mes transferts ce mois",
                "Historique par bénéficiaire"
            ]
        },
        
        IntentType.PAYMENT_REQUEST: {
            "template": """
Je ne peux pas traiter les paiements. Harena vous permet uniquement de consulter et analyser vos données bancaires.

**Pour payer vos factures :**
• Application mobile bancaire
• Espace client en ligne
• Prélèvement automatique
• RIB et virements manuels

**Je peux vous aider avec :**
• Consulter vos paiements récents
• Analyser vos factures par catégorie
• Voir l'évolution de vos charges

Voulez-vous analyser vos paiements récents ou vos dépenses par catégorie ?
            """.strip(),
            "suggestions": [
                "Mes paiements ce mois",
                "Analyse par type de facture",
                "Évolution de mes charges"
            ]
        },
        
        IntentType.CARD_OPERATIONS: {
            "template": """
Je ne peux pas effectuer d'opérations sur votre carte. Pour votre sécurité, ces actions nécessitent une authentification forte.

**Pour gérer votre carte :**
• Application bancaire (blocage immédiat)
• Numéro d'urgence de votre banque
• Espace client sécurisé
• Contactez votre agence

**Je peux vous aider avec :**
• Consulter vos dépenses par carte
• Analyser l'utilisation de vos cartes
• Voir les détails de vos cartes

Souhaitez-vous analyser vos dépenses par carte ou voir les informations de vos cartes ?
            """.strip(),
            "suggestions": [
                "Dépenses par carte ce mois",
                "Informations sur mes cartes",
                "Analyse utilisation carte"
            ]
        },
        
        IntentType.LOAN_REQUEST: {
            "template": """
Je ne peux pas traiter les demandes de crédit. Ces décisions nécessitent une étude personnalisée par votre conseiller.

**Pour votre demande de crédit :**
• Prenez rendez-vous avec votre conseiller
• Simulateur en ligne sur le site de votre banque
• Documentation nécessaire : revenus, charges, projet

**Je peux vous aider avec :**
• Analyser votre capacité d'épargne
• Voir vos crédits en cours
• Analyser vos revenus et charges

Voulez-vous analyser votre situation financière ou voir vos crédits actuels ?
            """.strip(),
            "suggestions": [
                "Analyser ma capacité d'épargne",
                "Voir mes crédits en cours",
                "Analyser revenus vs charges"
            ]
        },
        
        IntentType.ACCOUNT_MODIFICATION: {
            "template": """
Je ne peux pas modifier vos informations personnelles. Ces changements nécessitent une procédure sécurisée.

**Pour modifier vos coordonnées :**
• Connectez-vous à votre espace client
• Application mobile bancaire
• Contactez votre agence avec justificatifs
• Courrier sécurisé à votre banque

**Je peux vous aider avec :**
• Consulter vos informations de compte
• Voir l'historique de vos modifications
• Analyser vos données bancaires

Souhaitez-vous consulter les informations de vos comptes ?
            """.strip(),
            "suggestions": [
                "Informations de mes comptes",
                "Détails de mes produits",
                "Consulter mes coordonnées"
            ]
        },
        
        IntentType.INVESTMENT_OPERATIONS: {
            "template": """
Je ne peux pas effectuer d'opérations d'investissement. Ces transactions nécessitent des plateformes spécialisées.

**Pour vos investissements :**
• Plateforme de trading de votre banque
• Conseiller en investissement
• Applications spécialisées (boursorama, etc.)
• Espace client investissement

**Je peux vous aider avec :**
• Analyser vos revenus disponibles
• Voir vos comptes d'épargne
• Analyser votre capacité d'investissement

Voulez-vous analyser votre situation financière pour l'investissement ?
            """.strip(),
            "suggestions": [
                "Analyser ma capacité d'investissement",
                "Voir mes comptes d'épargne",
                "Revenus disponibles"
            ]
        }
    }
    
    # Generic template for unspecified unsupported actions
    GENERIC_UNSUPPORTED_TEMPLATE = {
        "template": """
Cette action n'est pas disponible via Harena. Je suis un assistant consultatif qui vous aide à comprendre et analyser vos finances.

**Pour les opérations bancaires :**
• Application mobile de votre banque
• Espace client en ligne
• Contactez votre conseiller bancaire
• Agence la plus proche

**Je peux vous aider avec :**
• Consulter vos comptes et transactions
• Analyser vos dépenses et revenus
• Comprendre vos produits bancaires
• Suivre l'évolution de vos finances

Que souhaitez-vous consulter ou analyser dans vos finances ?
        """.strip(),
        "suggestions": [
            "Voir mes comptes",
            "Analyser mes dépenses",
            "Comprendre mes produits"
        ]
    }
    
    # Out of scope template
    OUT_OF_SCOPE_TEMPLATE = {
        "template": """
Je suis spécialisé dans l'assistance bancaire et financière. Pour cette demande, je vous encourage à consulter des sources appropriées.

**Je peux vous aider avec vos finances :**
• Consulter vos comptes et soldes
• Analyser vos dépenses par catégorie
• Suivre l'évolution de vos finances
• Comprendre vos produits bancaires
• Rechercher des transactions spécifiques

Avez-vous une question sur vos finances que je puisse traiter ?
        """.strip(),
        "suggestions": [
            "Voir mon solde",
            "Analyser mes dépenses",
            "Mes dernières transactions"
        ]
    }
    
    @classmethod
    def get_redirection_response(cls, intent: IntentType) -> Dict[str, Any]:
        """Get appropriate redirection response for unsupported intent."""
        if intent in cls.REDIRECTION_TEMPLATES:
            return cls.REDIRECTION_TEMPLATES[intent]
        else:
            return cls.GENERIC_UNSUPPORTED_TEMPLATE
    
    @classmethod
    def format_unsupported_response(
        cls, 
        intent: IntentType, 
        user_message: str,
        detected_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Format complete response for unsupported action.
        
        Includes personalized elements based on the user's specific request.
        """
        base_response = cls.get_redirection_response(intent)
        
        # Add personalization based on detected patterns
        personalization = ""
        if "virement" in user_message.lower() or "transfer" in user_message.lower():
            personalization = "Concernant votre demande de virement, "
        elif "payer" in user_message.lower() or "facture" in user_message.lower():
            personalization = "Pour le paiement de votre facture, "
        elif "carte" in user_message.lower():
            personalization = "Pour votre carte bancaire, "
        
        formatted_response = personalization + base_response["template"]
        
        return {
            "response": formatted_response,
            "response_type": "redirection",
            "harena_limitation": True,
            "suggestions": base_response["suggestions"],
            "confidence": 0.9,
            "tone": "professional"
        }

# ================================
# MAIN INTENT TAXONOMY CLASS
# ================================

class IntentTaxonomy:
    """
    Complete intent taxonomy for Harena conversation service.
    
    Central class that orchestrates all intent classification logic,
    from keyword-based fallbacks to sophisticated action detection
    and appropriate response generation.
    """
    
    # Primary categories with detailed definitions
    PRIMARY_CATEGORIES = {
        "CONSULTATION": {
            "description": "Information requests about account status and data (Harena Supported)",
            "sub_categories": [
                IntentType.BALANCE_INQUIRY,
                IntentType.ACCOUNT_OVERVIEW,
                IntentType.TRANSACTION_SEARCH,
                IntentType.TRANSACTION_DETAILS,
                IntentType.STATEMENT_REQUEST
            ],
            "examples": [
                "Quel est mon solde ?",
                "Mes dernières transactions",
                "Mes achats Amazon",
                "Détails de cette transaction de 50€",
                "Mon relevé du mois dernier"
            ],
            "confidence_range": (0.5, 0.98),
            "processing_priority": "high"
        },
        
        "ANALYSIS": {
            "description": "Financial analysis and insights (Harena Supported)",
            "sub_categories": [
                IntentType.SPENDING_ANALYSIS,
                IntentType.CATEGORY_ANALYSIS,
                IntentType.MERCHANT_ANALYSIS,
                IntentType.TEMPORAL_ANALYSIS,
                IntentType.BUDGET_ANALYSIS,
                IntentType.TREND_ANALYSIS,
                IntentType.INCOME_ANALYSIS,
                IntentType.COMPARISON_ANALYSIS
            ],
            "examples": [
                "Mes dépenses ce mois",
                "Combien j'ai dépensé en restaurant ?",
                "Mes achats chez Carrefour cette année",
                "Évolution de mes dépenses transport",
                "Mon budget shopping vs réalisé",
                "Tendance de mes dépenses",
                "Mes revenus ce trimestre",
                "Comparer janvier vs février"
            ],
            "confidence_range": (0.5, 0.95),
            "processing_priority": "high"
        },
        
        "INFORMATION_SUPPORT": {
            "description": "General information and support (Harena Supported)",
            "sub_categories": [
                IntentType.ACCOUNT_INFORMATION,
                IntentType.PRODUCT_INFORMATION,
                IntentType.FEE_INQUIRY,
                IntentType.GENERAL_INQUIRY,
                IntentType.HELP_REQUEST
            ],
            "examples": [
                "Comment fonctionne mon compte courant ?",
                "Qu'est-ce qu'un livret A ?",
                "Quels sont les frais de ma carte ?",
                "Comment lire mon relevé ?",
                "J'ai besoin d'aide pour comprendre"
            ],
            "confidence_range": (0.5, 0.9),
            "processing_priority": "medium"
        },
        
        "CONVERSATIONAL": {
            "description": "Standard conversational interactions",
            "sub_categories": [
                IntentType.GREETING,
                IntentType.GOODBYE,
                IntentType.THANKS,
                IntentType.CLARIFICATION_REQUEST,
                IntentType.POLITENESS
            ],
            "examples": [
                "Bonjour", "Salut", "Hello",
                "Au revoir", "À bientôt",
                "Merci", "Merci beaucoup",
                "Pouvez-vous répéter ?", "Je n'ai pas compris",
                "Comment allez-vous ?", "Ça va ?"
            ],
            "confidence_range": (0.7, 0.99),
            "processing_priority": "low"
        },
        
        "UNSUPPORTED_ACTIONS": {
            "description": "Actions not supported by Harena - Redirection required",
            "sub_categories": [
                IntentType.TRANSFER_REQUEST,
                IntentType.PAYMENT_REQUEST,
                IntentType.CARD_OPERATIONS,
                IntentType.LOAN_REQUEST,
                IntentType.ACCOUNT_MODIFICATION,
                IntentType.INVESTMENT_OPERATIONS,
                IntentType.UNSUPPORTED_ACTION
            ],
            "examples": [
                "Faire un virement de 500€",
                "Payer ma facture EDF",
                "Bloquer ma carte",
                "Demande de crédit auto",
                "Changer mon adresse",
                "Acheter des actions"
            ],
            "confidence_range": (0.8, 0.98),
            "processing_priority": "high",  # High priority for proper redirection
            "response_strategy": "redirect_with_explanation"
        },
        
        "UNRESOLVED": {
            "description": "Unidentifiable or out-of-scope requests",
            "sub_categories": [
                IntentType.OUT_OF_SCOPE,
                IntentType.UNKNOWN,
                IntentType.AMBIGUOUS,
                IntentType.INSUFFICIENT_CONTEXT
            ],
            "examples": [
                "Quel temps fait-il ?",
                "kdsjfkld", "???",
                "Ça", "Le truc",
                "Par rapport à hier (sans contexte)",
                "Je veux..."
            ],
            "confidence_range": (0.0, 0.5),
            "processing_priority": "low",
            "response_strategy": "clarification_request"
        }
    }
    
    @classmethod
    def classify_intent(
        cls, 
        user_message: str, 
        use_fallback: bool = True
    ) -> Tuple[IntentType, float, Dict[str, Any]]:
        """
        Classify user intent using multiple strategies.
        
        Args:
            user_message: User input text
            use_fallback: Whether to use keyword-based fallback
            
        Returns:
            Tuple of (intent, confidence, metadata)
        """
        metadata = {
            "classification_method": "hybrid",
            "detected_patterns": [],
            "alternatives": [],
            "processing_notes": []
        }
        
        # First: Check for unsupported actions (high precision required)
        is_action, action_confidence, action_patterns = UnsupportedActionDetector.detect_unsupported_action(user_message)
        
        if is_action:
            action_type = UnsupportedActionDetector.categorize_action_type(user_message)
            metadata["classification_method"] = "unsupported_action_detection"
            metadata["detected_patterns"] = action_patterns
            metadata["processing_notes"].append("High-confidence unsupported action detected")
            
            return action_type, action_confidence, metadata
        
        # Second: Keyword-based classification
        if use_fallback:
            keyword_candidates = IntentClassificationMatrix.classify_by_keywords(user_message)
            
            if keyword_candidates:
                best_intent, best_confidence = keyword_candidates[0]
                
                # Validate confidence threshold
                min_threshold = cls._get_minimum_confidence(best_intent)
                if best_confidence >= min_threshold:
                    metadata["classification_method"] = "keyword_based"
                    metadata["alternatives"] = keyword_candidates[1:3]  # Top 2 alternatives
                    metadata["processing_notes"].append(f"Keyword-based classification with {len(keyword_candidates)} candidates")
                    
                    return best_intent, best_confidence, metadata
        
        # Third: Default to unknown with low confidence
        metadata["classification_method"] = "default_unknown"
        metadata["processing_notes"].append("No clear classification pattern found")
        
        return IntentType.UNKNOWN, 0.2, metadata
    
    @classmethod
    def _get_minimum_confidence(cls, intent: IntentType) -> float:
        """Get minimum confidence threshold for intent type."""
        confidence_map = {
            # Consultation intents
            IntentType.BALANCE_INQUIRY: ConfidenceThresholds.CONSULTATION_MIN,
            IntentType.ACCOUNT_OVERVIEW: ConfidenceThresholds.CONSULTATION_MIN,
            IntentType.TRANSACTION_SEARCH: ConfidenceThresholds.CONSULTATION_MIN,
            
            # Analysis intents
            IntentType.SPENDING_ANALYSIS: ConfidenceThresholds.ANALYSIS_MIN,
            IntentType.CATEGORY_ANALYSIS: ConfidenceThresholds.ANALYSIS_MIN,
            IntentType.MERCHANT_ANALYSIS: ConfidenceThresholds.ANALYSIS_MIN,
            
            # Conversational
            IntentType.GREETING: ConfidenceThresholds.CONVERSATIONAL_MIN,
            IntentType.GOODBYE: ConfidenceThresholds.CONVERSATIONAL_MIN,
            IntentType.THANKS: ConfidenceThresholds.CONVERSATIONAL_MIN,
            
            # Unsupported actions
            IntentType.TRANSFER_REQUEST: ConfidenceThresholds.UNSUPPORTED_ACTION_MIN,
            IntentType.PAYMENT_REQUEST: ConfidenceThresholds.UNSUPPORTED_ACTION_MIN,
            IntentType.CARD_OPERATIONS: ConfidenceThresholds.UNSUPPORTED_ACTION_MIN,
        }
        
        return confidence_map.get(intent, 0.5)  # Default minimum
    
    @classmethod
    def is_supported_by_harena(cls, intent: IntentType) -> bool:
        """Check if intent is supported by Harena's consultation scope."""
        supported_categories = {
            IntentType.BALANCE_INQUIRY, IntentType.ACCOUNT_OVERVIEW,
            IntentType.TRANSACTION_SEARCH, IntentType.TRANSACTION_DETAILS,
            IntentType.STATEMENT_REQUEST,
            IntentType.SPENDING_ANALYSIS, IntentType.CATEGORY_ANALYSIS,
            IntentType.MERCHANT_ANALYSIS, IntentType.TEMPORAL_ANALYSIS,
            IntentType.BUDGET_ANALYSIS, IntentType.TREND_ANALYSIS,
            IntentType.INCOME_ANALYSIS, IntentType.COMPARISON_ANALYSIS,
            IntentType.ACCOUNT_INFORMATION, IntentType.PRODUCT_INFORMATION,
            IntentType.FEE_INQUIRY, IntentType.GENERAL_INQUIRY,
            IntentType.HELP_REQUEST, IntentType.GREETING, IntentType.GOODBYE,
            IntentType.THANKS, IntentType.CLARIFICATION_REQUEST, IntentType.POLITENESS
        }
        return intent in supported_categories
    
    @classmethod
    def get_response_strategy(cls, intent: IntentType) -> str:
        """Get appropriate response strategy for intent."""
        if cls.is_supported_by_harena(intent):
            return "process_normally"
        elif intent in {
            IntentType.TRANSFER_REQUEST, IntentType.PAYMENT_REQUEST,
            IntentType.CARD_OPERATIONS, IntentType.LOAN_REQUEST,
            IntentType.ACCOUNT_MODIFICATION, IntentType.INVESTMENT_OPERATIONS,
            IntentType.UNSUPPORTED_ACTION
        }:
            return "redirect_with_explanation"
        elif intent == IntentType.OUT_OF_SCOPE:
            return "polite_redirection"
        else:
            return "clarification_request"
    
    @classmethod
    def get_example_clarification_questions(cls, intent: IntentType) -> List[str]:
        """Get example questions to help clarify ambiguous intents."""
        clarification_map = {
            IntentType.AMBIGUOUS: [
                "Pouvez-vous préciser votre demande ?",
                "Souhaitez-vous consulter vos comptes ou analyser vos dépenses ?",
                "Cherchez-vous des informations sur un produit spécifique ?"
            ],
            IntentType.UNKNOWN: [
                "Je peux vous aider avec vos finances. Que souhaitez-vous savoir ?",
                "Voulez-vous voir votre solde, vos transactions, ou analyser vos dépenses ?",
                "Avez-vous une question sur vos comptes ou vos produits bancaires ?"
            ],
            IntentType.INSUFFICIENT_CONTEXT: [
                "Pouvez-vous donner plus de détails sur votre demande ?",
                "À quoi faites-vous référence exactement ?",
                "Quelle information précise cherchez-vous ?"
            ],
            IntentType.OUT_OF_SCOPE: [
                "Pour les questions bancaires, que puis-je vous expliquer ?",
                "Souhaitez-vous consulter vos comptes ou comprendre un produit bancaire ?",
                "Je peux vous aider avec vos finances - avez-vous une question spécifique ?"
            ]
        }
        
        return clarification_map.get(intent, [
            "Comment puis-je vous aider avec vos finances ?",
            "Que souhaitez-vous consulter ou analyser ?"
        ])
    
    @classmethod
    def validate_classification_result(
        cls,
        intent: IntentType,
        confidence: float,
        user_message: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate classification result for consistency and quality.
        
        Returns:
            - is_valid: Whether classification is acceptable
            - warnings: List of validation warnings
        """
        warnings = []
        is_valid = True
        
        # Check confidence thresholds
        min_confidence = cls._get_minimum_confidence(intent)
        if confidence < min_confidence:
            warnings.append(f"Confidence {confidence:.2f} below minimum {min_confidence:.2f} for {intent}")
            is_valid = False
        
        # Check for misclassified unsupported actions
        if intent not in {
            IntentType.TRANSFER_REQUEST, IntentType.PAYMENT_REQUEST,
            IntentType.CARD_OPERATIONS, IntentType.UNSUPPORTED_ACTION
        }:
            action_keywords = ['virement', 'payer', 'bloquer', 'transférer', 'faire un']
            if any(keyword in user_message.lower() for keyword in action_keywords):
                warnings.append("Possible unsupported action classified as supported intent")
        
        # Check for over-confident unknown classifications
        if intent == IntentType.UNKNOWN and confidence > 0.5:
            warnings.append("Unknown intent should have low confidence")
            is_valid = False
        
        return is_valid, warnings