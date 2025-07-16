import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.conversation import (
    FinancialIntent, 
    IntentResult, 
    EntityHints, 
    ProcessingError
)
from ..clients.deepseek_client import deepseek_client, DeepSeekError
from ..config.settings import settings

logger = logging.getLogger(__name__)

class IntentClassifierAgent:
    """Agent de classification d'intentions financières avec DeepSeek"""
    
    def __init__(self):
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD
        self.model_name = settings.DEEPSEEK_CHAT_MODEL
        
        # Métriques de l'agent
        self._metrics = {
            "total_classifications": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "unclear_intents": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0,
            "intent_distribution": {},
            "last_classification_time": None
        }
        
        logger.info(f"Agent classification initialisé - Seuil confiance: {self.confidence_threshold}")
    
    def _build_classification_prompt(self, user_message: str) -> List[Dict[str, str]]:
        """Construit le prompt few-shot pour la classification"""
        
        system_prompt = """Tu es un expert en classification d'intentions financières pour une application de gestion de finances personnelles.

INSTRUCTIONS:
1. Analyse le message utilisateur et classifie son intention
2. Extrais les entités pertinentes (marchands, catégories, montants, dates)
3. Évalue le niveau de confiance de ta classification
4. Réponds UNIQUEMENT en JSON valide, sans texte additionnel

INTENTIONS SUPPORTÉES:
- search_by_merchant: Recherche par nom de marchand ("mes achats netflix", "transactions amazon")
- search_by_category: Recherche par catégorie ("mes restaurants", "mes courses")
- search_by_amount: Recherche par montant ("plus de 100€", "moins de 50€")
- search_by_date: Recherche par date/période ("janvier 2024", "ce mois", "cette semaine")
- search_general: Recherche générale ("mes transactions", "historique")
- spending_analysis: Analyse des dépenses ("combien j'ai dépensé", "analyse budget")
- income_analysis: Analyse des revenus ("mes revenus", "salaire")
- unclear_intent: Intention non claire ou ambiguë

ENTITÉS À EXTRAIRE:
- merchant: Nom du marchand (netflix, amazon, uber, etc.)
- category: Catégorie (restaurant, transport, courses, etc.)
- amount: Montant numérique
- operator: Opérateur (gt=plus de, lt=moins de, eq=égal à)
- period: Période (ce mois, cette semaine, etc.)
- date: Date spécifique

FORMAT DE RÉPONSE (JSON uniquement):
{
    "intent": "nom_intention",
    "confidence": 0.95,
    "entities": {
        "merchant": "nom_marchand",
        "category": "nom_catégorie",
        "amount": "montant",
        "operator": "opérateur",
        "period": "période",
        "date": "date"
    },
    "reasoning": "Explication brève du raisonnement"
}"""

        examples = [
            {
                "role": "user",
                "content": "mes achats netflix"
            },
            {
                "role": "assistant", 
                "content": '{"intent": "search_by_merchant", "confidence": 0.95, "entities": {"merchant": "netflix"}, "reasoning": "Recherche explicite par nom de marchand Netflix"}'
            },
            {
                "role": "user",
                "content": "mes restaurants ce mois"
            },
            {
                "role": "assistant",
                "content": '{"intent": "search_by_category", "confidence": 0.92, "entities": {"category": "restaurant", "period": "ce mois"}, "reasoning": "Recherche par catégorie restaurant avec période temporelle"}'
            },
            {
                "role": "user",
                "content": "plus de 100 euros"
            },
            {
                "role": "assistant",
                "content": '{"intent": "search_by_amount", "confidence": 0.88, "entities": {"amount": "100", "operator": "gt"}, "reasoning": "Recherche par montant avec opérateur supérieur"}'
            },
            {
                "role": "user",
                "content": "combien j'ai dépensé en janvier"
            },
            {
                "role": "assistant",
                "content": '{"intent": "spending_analysis", "confidence": 0.90, "entities": {"period": "janvier"}, "reasoning": "Demande d\'analyse des dépenses pour une période spécifique"}'
            },
            {
                "role": "user",
                "content": "salut"
            },
            {
                "role": "assistant",
                "content": '{"intent": "unclear_intent", "confidence": 0.10, "entities": {}, "reasoning": "Message de salutation sans intention financière claire"}'
            }
        ]
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(examples)
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _parse_response(self, response_content: str) -> Dict[str, Any]:
        """Parse et valide la réponse JSON de DeepSeek"""
        try:
            # Nettoyer la réponse (enlever markdown si présent)
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            # Parser le JSON
            result = json.loads(content)
            
            # Validation des champs requis
            if "intent" not in result:
                raise ValueError("Champ 'intent' manquant dans la réponse")
            if "confidence" not in result:
                raise ValueError("Champ 'confidence' manquant dans la réponse")
            
            # Validation de l'intention
            if result["intent"] not in [intent.value for intent in FinancialIntent]:
                logger.warning(f"Intention non reconnue: {result['intent']}, fallback vers unclear_intent")
                result["intent"] = FinancialIntent.UNCLEAR_INTENT.value
                result["confidence"] = 0.1
            
            # Validation de la confiance
            confidence = float(result["confidence"])
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(f"Confiance invalide: {confidence}")
            
            # Entités par défaut si manquantes
            if "entities" not in result:
                result["entities"] = {}
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}, contenu: {response_content}")
            raise ProcessingError(
                message=f"Erreur parsing JSON: {str(e)}",
                details={"response_content": response_content}
            )
        except Exception as e:
            logger.error(f"Erreur validation réponse: {e}")
            raise ProcessingError(
                message=f"Erreur validation: {str(e)}",
                details={"response_content": response_content}
            )
    
    def _handle_unclear_intent(self, result: Dict[str, Any]) -> IntentResult:
        """Gère les intentions peu claires"""
        logger.info(f"Intention peu claire détectée - Confiance: {result['confidence']}")
        
        return IntentResult(
            intent=FinancialIntent.UNCLEAR_INTENT,
            confidence=result["confidence"],
            entities=EntityHints(**result.get("entities", {})),
            reasoning=f"Confiance insuffisante ({result['confidence']:.2f} < {self.confidence_threshold}). " + 
                     result.get("reasoning", "")
        )
    
    def _create_intent_result(self, parsed_result: Dict[str, Any]) -> IntentResult:
        """Crée un IntentResult à partir du résultat parsé"""
        return IntentResult(
            intent=FinancialIntent(parsed_result["intent"]),
            confidence=parsed_result["confidence"],
            entities=EntityHints(**parsed_result.get("entities", {})),
            reasoning=parsed_result.get("reasoning", "")
        )
    
    def _update_metrics(self, success: bool, processing_time: float, intent: FinancialIntent, confidence: float):
        """Met à jour les métriques de l'agent"""
        self._metrics["total_classifications"] += 1
        self._metrics["last_classification_time"] = datetime.utcnow()
        
        if success:
            self._metrics["successful_classifications"] += 1
            
            # Mise à jour moyenne confiance
            total_successful = self._metrics["successful_classifications"]
            current_avg = self._metrics["avg_confidence"]
            self._metrics["avg_confidence"] = (
                (current_avg * (total_successful - 1) + confidence) / total_successful
            )
            
            # Distribution des intentions
            intent_str = intent.value
            if intent_str not in self._metrics["intent_distribution"]:
                self._metrics["intent_distribution"][intent_str] = 0
            self._metrics["intent_distribution"][intent_str] += 1
            
            # Compteur unclear_intent
            if intent == FinancialIntent.UNCLEAR_INTENT:
                self._metrics["unclear_intents"] += 1
        else:
            self._metrics["failed_classifications"] += 1
        
        # Mise à jour temps de traitement moyen
        total_classifications = self._metrics["total_classifications"]
        current_avg_time = self._metrics["avg_processing_time"]
        self._metrics["avg_processing_time"] = (
            (current_avg_time * (total_classifications - 1) + processing_time) / total_classifications
        )
    
    async def classify_intent(self, user_message: str) -> IntentResult:
        """
        Classifie l'intention d'un message utilisateur
        
        Args:
            user_message: Message de l'utilisateur
            
        Returns:
            IntentResult: Résultat de la classification
        """
        start_time = time.time()
        
        try:
            # Validation du message
            if not user_message or not user_message.strip():
                raise ProcessingError(
                    message="Message utilisateur vide",
                    details={"user_message": user_message}
                )
            
            # Construction du prompt
            messages = self._build_classification_prompt(user_message.strip())
            
            # Appel à DeepSeek
            logger.debug(f"Classification pour: '{user_message[:50]}...'")
            response = await deepseek_client.chat_completion(messages)
            
            # Parsing de la réponse
            parsed_result = self._parse_response(response.content)
            
            # Vérification du seuil de confiance
            if parsed_result["confidence"] < self.confidence_threshold:
                result = self._handle_unclear_intent(parsed_result)
            else:
                result = self._create_intent_result(parsed_result)
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self._update_metrics(True, processing_time, result.intent, result.confidence)
            
            logger.info(f"Classification réussie: {result.intent.value} (confiance: {result.confidence:.2f})")
            
            return result
            
        except DeepSeekError as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time, FinancialIntent.UNCLEAR_INTENT, 0.0)
            
            logger.error(f"Erreur DeepSeek: {e.message}")
            
            # Fallback gracieux
            return IntentResult(
                intent=FinancialIntent.UNCLEAR_INTENT,
                confidence=0.1,
                entities=EntityHints(),
                reasoning=f"Erreur DeepSeek: {e.message}"
            )
            
        except ProcessingError as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time, FinancialIntent.UNCLEAR_INTENT, 0.0)
            
            logger.error(f"Erreur de traitement: {e.message}")
            raise
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time, FinancialIntent.UNCLEAR_INTENT, 0.0)
            
            logger.error(f"Erreur inattendue: {str(e)}")
            
            # Fallback gracieux
            return IntentResult(
                intent=FinancialIntent.UNCLEAR_INTENT,
                confidence=0.1,
                entities=EntityHints(),
                reasoning=f"Erreur système: {str(e)}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'agent"""
        total_classifications = self._metrics["total_classifications"]
        
        return {
            "total_classifications": total_classifications,
            "successful_classifications": self._metrics["successful_classifications"],
            "failed_classifications": self._metrics["failed_classifications"],
            "success_rate": (
                self._metrics["successful_classifications"] / total_classifications 
                if total_classifications > 0 else 0
            ),
            "unclear_intents": self._metrics["unclear_intents"],
            "unclear_rate": (
                self._metrics["unclear_intents"] / total_classifications 
                if total_classifications > 0 else 0
            ),
            "avg_confidence": self._metrics["avg_confidence"],
            "avg_processing_time": self._metrics["avg_processing_time"],
            "intent_distribution": self._metrics["intent_distribution"],
            "last_classification_time": self._metrics["last_classification_time"]
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        self._metrics = {
            "total_classifications": 0,
            "successful_classifications": 0,
            "failed_classifications": 0,
            "unclear_intents": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0,
            "intent_distribution": {},
            "last_classification_time": None
        }
        logger.info("Métriques de l'agent remises à zéro")

# Instance globale de l'agent
intent_classifier = IntentClassifierAgent()