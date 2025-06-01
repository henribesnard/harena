"""
Détecteur d'intention utilisant DeepSeek Reasoner.

Ce module analyse les messages utilisateur pour déterminer leur intention
et extraire les paramètres pertinents pour la recherche.
"""
import logging
import time
from typing import Dict, Any, List, Optional
import re

from conversation_service.models import IntentType, DetectedIntent
from conversation_service.core.deepseek_client import deepseek_client

logger = logging.getLogger(__name__)


class IntentDetector:
    """Détecteur d'intention principal."""
    
    def __init__(self):
        self.cache = {}  # Cache simple pour les intentions récentes
        self.max_cache_size = 100
        self._initialized = False
        
        # Patterns pour détection rapide
        self.quick_patterns = {
            IntentType.GREETING: [
                r'\b(bonjour|salut|hello|hi|bonsoir)\b',
                r'\b(comment ça va|ça va)\b'
            ],
            IntentType.HELP: [
                r'\b(aide|help|comment|que faire)\b',
                r'\b(comment utiliser|comment faire)\b'
            ],
            IntentType.SEARCH_TRANSACTIONS: [
                r'\b(trouver|rechercher|voir|afficher).*transaction',
                r'\btransaction.*\b(du|de|le|la|ce|cette)',
                r'\b(show|find|get).*transaction'
            ],
            IntentType.SPENDING_ANALYSIS: [
                r'\b(combien|montant|total|somme).*dépens',
                r'\bdépens.*\b(en|pour|chez|au)',
                r'\b(coût|prix|facture).*total'
            ],
            IntentType.ACCOUNT_SUMMARY: [
                r'\b(solde|compte|balance)\b',
                r'\bétat.*compte',
                r'\bsituation.*financière'
            ]
        }
        
    async def initialize(self):
        """Initialise le détecteur d'intention."""
        self._initialized = True
        logger.info("IntentDetector initialisé")
    
    def is_initialized(self) -> bool:
        """Vérifie si le détecteur est initialisé."""
        return self._initialized
    
    async def detect_intent(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> DetectedIntent:
        """
        Détecte l'intention d'un message utilisateur.
        
        Args:
            user_message: Message de l'utilisateur
            conversation_history: Historique de conversation
            user_context: Contexte utilisateur
            
        Returns:
            DetectedIntent: Intention détectée
        """
        if not user_message.strip():
            return DetectedIntent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                parameters={},
                reasoning="Message vide"
            )
        
        # Vérifier le cache
        cache_key = self._get_cache_key(user_message, conversation_history)
        if cache_key in self.cache:
            logger.debug(f"Cache hit pour détection d'intention: {user_message[:50]}...")
            return self.cache[cache_key]
        
        # Détection rapide par patterns
        quick_intent = self._quick_pattern_detection(user_message)
        if quick_intent and quick_intent.confidence > 0.8:
            self._cache_intent(cache_key, quick_intent)
            return quick_intent
        
        # Détection avancée via DeepSeek
        advanced_intent = await self._advanced_intent_detection(
            user_message, conversation_history, user_context
        )
        
        # Combiner les résultats si nécessaire
        final_intent = self._combine_intents(quick_intent, advanced_intent)
        
        # Mettre en cache
        self._cache_intent(cache_key, final_intent)
        
        return final_intent
    
    def _quick_pattern_detection(self, message: str) -> Optional[DetectedIntent]:
        """
        Détection rapide par patterns regex.
        
        Args:
            message: Message à analyser
            
        Returns:
            DetectedIntent ou None si pas de match
        """
        message_lower = message.lower()
        
        for intent_type, patterns in self.quick_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    # Extraire des paramètres basiques
                    parameters = self._extract_basic_parameters(message_lower, intent_type)
                    
                    return DetectedIntent(
                        intent_type=intent_type,
                        confidence=0.85,
                        parameters=parameters,
                        reasoning=f"Pattern match: {pattern}"
                    )
        
        return None
    
    def _extract_basic_parameters(self, message: str, intent_type: IntentType) -> Dict[str, Any]:
        """
        Extrait des paramètres basiques selon le type d'intention.
        
        Args:
            message: Message en minuscules
            intent_type: Type d'intention
            
        Returns:
            Dict: Paramètres extraits
        """
        parameters = {}
        
        # Extraction de périodes temporelles
        time_patterns = {
            'aujourd\'hui': 'today',
            'hier': 'yesterday',
            'cette semaine': 'this_week',
            'la semaine dernière': 'last_week',
            'ce mois': 'this_month',
            'le mois dernier': 'last_month',
            'cette année': 'this_year',
            'l\'année dernière': 'last_year'
        }
        
        for pattern, period in time_patterns.items():
            if pattern in message:
                parameters['time_period'] = period
                break
        
        # Extraction de montants
        amount_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:€|euro|eur)', message)
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '.')
            try:
                parameters['amount'] = float(amount_str)
            except ValueError:
                pass
        
        # Extraction de catégories/marchands communs
        category_patterns = {
            'restaurant': ['restaurant', 'resto', 'restau'],
            'supermarché': ['supermarché', 'course', 'monoprix', 'carrefour'],
            'carburant': ['carburant', 'essence', 'station'],
            'transport': ['transport', 'metro', 'bus', 'train'],
            'santé': ['pharmacie', 'médecin', 'dentiste'],
            'shopping': ['shopping', 'vêtement', 'magasin']
        }
        
        for category, keywords in category_patterns.items():
            if any(keyword in message for keyword in keywords):
                parameters['category'] = category
                break
        
        return parameters
    
    async def _advanced_intent_detection(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> DetectedIntent:
        """
        Détection avancée via DeepSeek Reasoner.
        
        Args:
            user_message: Message utilisateur
            conversation_history: Historique de conversation
            user_context: Contexte utilisateur
            
        Returns:
            DetectedIntent: Intention détectée
        """
        try:
            # Utiliser le client DeepSeek
            intent_result = await deepseek_client.detect_intent(
                user_message, conversation_history
            )
            
            # Convertir en DetectedIntent
            return DetectedIntent(
                intent_type=IntentType(intent_result.get("intent_type", "unknown")),
                confidence=intent_result.get("confidence", 0.5),
                parameters=intent_result.get("parameters", {}),
                reasoning=intent_result.get("reasoning", "Détection DeepSeek")
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection avancée: {e}")
            return DetectedIntent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.3,
                parameters={},
                reasoning=f"Erreur détection avancée: {str(e)}"
            )
    
    def _combine_intents(
        self,
        quick_intent: Optional[DetectedIntent],
        advanced_intent: DetectedIntent
    ) -> DetectedIntent:
        """
        Combine les résultats de détection rapide et avancée.
        
        Args:
            quick_intent: Intention détectée rapidement
            advanced_intent: Intention détectée par IA
            
        Returns:
            DetectedIntent: Intention finale
        """
        if not quick_intent:
            return advanced_intent
        
        # Si les deux détections concordent, augmenter la confiance
        if quick_intent.intent_type == advanced_intent.intent_type:
            return DetectedIntent(
                intent_type=quick_intent.intent_type,
                confidence=min(0.95, quick_intent.confidence + advanced_intent.confidence * 0.3),
                parameters={**quick_intent.parameters, **advanced_intent.parameters},
                reasoning=f"Pattern + IA: {quick_intent.reasoning} | {advanced_intent.reasoning}"
            )
        
        # Si conflit, prendre la plus fiable
        if quick_intent.confidence > advanced_intent.confidence:
            return quick_intent
        else:
            return advanced_intent
    
    def _get_cache_key(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Génère une clé de cache pour un message."""
        # Simplifier le message pour le cache
        import hashlib
        
        message_hash = hashlib.md5(message.lower().encode()).hexdigest()[:8]
        
        # Ajouter le contexte de l'historique récent
        if history and len(history) > 0:
            last_message = history[-1].get("content", "")
            context_hash = hashlib.md5(last_message.encode()).hexdigest()[:4]
            return f"{message_hash}_{context_hash}"
        
        return message_hash
    
    def _cache_intent(self, key: str, intent: DetectedIntent):
        """Met en cache une intention détectée."""
        # Limiter la taille du cache
        if len(self.cache) >= self.max_cache_size:
            # Supprimer le plus ancien
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = intent
    
    def get_intent_suggestions(self, partial_message: str) -> List[str]:
        """
        Obtient des suggestions d'intention pour un message partiel.
        
        Args:
            partial_message: Message partiel
            
        Returns:
            List[str]: Suggestions de complétion
        """
        suggestions = []
        message_lower = partial_message.lower()
        
        # Suggestions basées sur les mots-clés
        if "dépens" in message_lower:
            suggestions.extend([
                "Combien j'ai dépensé ce mois-ci ?",
                "Mes dépenses en restaurant",
                "Total des dépenses cette semaine"
            ])
        
        if "transaction" in message_lower:
            suggestions.extend([
                "Mes dernières transactions",
                "Transactions de plus de 50€",
                "Transactions chez Monoprix"
            ])
        
        if "compte" in message_lower:
            suggestions.extend([
                "Solde de mon compte",
                "État de mes comptes",
                "Résumé financier"
            ])
        
        return suggestions[:5]  # Limiter à 5 suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du détecteur."""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "patterns_count": sum(len(patterns) for patterns in self.quick_patterns.values()),
            "supported_intents": [intent.value for intent in IntentType]
        }
    
    def clear_cache(self):
        """Vide le cache des intentions."""
        self.cache.clear()
        logger.info("Cache des intentions vidé")


# Instance globale
intent_detector = IntentDetector()