"""
Classification des intentions utilisateur.

Ce module fournit la logique pour analyser les requêtes utilisateur
et déterminer leur intention, en utilisant des règles et le LLM.
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config.logging import get_logger
from ..config.constants import INTENT_TYPES
from ..llm.llm_service import LLMService
from ..models.intent import IntentType, IntentClassification
from ..llm.prompt_templates import get_intent_classification_prompt

logger = get_logger(__name__)


class IntentClassifier:
    """
    Classificateur d'intention pour les requêtes utilisateur.
    
    Cette classe analyse les requêtes utilisateur pour déterminer
    l'intention sous-jacente, en utilisant une combinaison de règles
    et de LLM.
    """
    
    def __init__(self, llm_service: LLMService):
        """
        Initialise le classificateur d'intention.
        
        Args:
            llm_service: Service LLM pour la classification par IA
        """
        self.llm_service = llm_service
        
        # Compiler les patterns de règles pour la détection rapide
        self.rule_patterns = {
            IntentType.CHECK_BALANCE: [
                r'\bsolde\b', r'\bbalance\b', r'\bargent\b.*\bcompte\b',
                r'\bcombien\b.*\bsur mon compte\b', r'\bcompte\b.*\bcontient\b'
            ],
            IntentType.SEARCH_TRANSACTION: [
                r'\btransaction\b', r'\bdépense\b', r'\bachat\b', r'\bpaiement\b',
                r'\bcombien\b.*\bdépensé\b', r'\btrouve\b.*\bpaiement\b'
            ],
            IntentType.ANALYZE_SPENDING: [
                r'\banalyse\b', r'\brépartition\b', r'\bcatégorie\b',
                r'\bdépenses\b.*\bcatégories\b', r'\banalyser\b.*\bdépenses\b'
            ],
            IntentType.ACCOUNT_INFO: [
                r'\bcomptes\b', r'\bmes comptes\b', r'\binformation\b.*\bcompte\b',
                r'\bliste\b.*\bcomptes\b', r'\bquels comptes\b'
            ],
            IntentType.HELP: [
                r'\baide\b', r'\bhelp\b', r'\bcomment\b.*\butiliser\b',
                r'\bque peux-tu faire\b', r'\bfonctionnalités\b'
            ]
        }
        
        # Compiler les patterns
        self.compiled_patterns = {}
        for intent, patterns in self.rule_patterns.items():
            self.compiled_patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        logger.info("Classificateur d'intention initialisé")
    
    async def classify_intent(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> IntentClassification:
        """
        Détermine l'intention d'une requête utilisateur.
        
        Args:
            query: Requête utilisateur à analyser
            conversation_context: Contexte de conversation optionnel
            
        Returns:
            Classification d'intention
        """
        logger.info(f"Classification de l'intention pour la requête: {query}")
        
        # Tentative de classification par règles
        rule_classification = self._classify_by_rules(query)
        if rule_classification and rule_classification.confidence > 0.8:
            logger.info(f"Intention classifiée par règles: {rule_classification.intent} (confiance: {rule_classification.confidence})")
            return rule_classification
        
        # Classification par LLM
        try:
            llm_classification = await self._classify_by_llm(query, conversation_context)
            logger.info(f"Intention classifiée par LLM: {llm_classification.intent} (confiance: {llm_classification.confidence})")
            return llm_classification
        except Exception as e:
            logger.error(f"Erreur lors de la classification par LLM: {str(e)}")
            
            # En cas d'échec du LLM, utiliser la classification par règles si disponible
            if rule_classification:
                return rule_classification
            
            # Sinon, retourner une classification par défaut
            return IntentClassification(
                intent=IntentType.GENERAL_QUERY,
                confidence=0.5,
                entities={}
            )
    
    def _classify_by_rules(self, query: str) -> Optional[IntentClassification]:
        """
        Classifie l'intention en utilisant des règles prédéfinies.
        
        Args:
            query: Requête à classifier
            
        Returns:
            Classification d'intention ou None si aucune règle ne correspond
        """
        # Vérifier chaque pattern pour chaque intention
        matches = {}
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    if intent not in matches:
                        matches[intent] = 0
                    matches[intent] += 1
        
        # Si pas de correspondance, retourner None
        if not matches:
            return None
        
        # Trouver l'intention avec le plus de correspondances
        best_intent = max(matches.items(), key=lambda x: x[1])
        intent_type = best_intent[0]
        match_count = best_intent[1]
        
        # Calculer un score de confiance basé sur le nombre de correspondances
        # et la longueur de la requête (heuristique simple)
        confidence = min(0.5 + (match_count * 0.1), 0.9)
        
        # Extraire les entités basiques (dates, montants, etc.)
        entities = self._extract_basic_entities(query)
        
        return IntentClassification(
            intent=intent_type,
            confidence=confidence,
            entities=entities
        )
    
    async def _classify_by_llm(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> IntentClassification:
        """
        Classifie l'intention en utilisant le LLM.
        
        Args:
            query: Requête à classifier
            conversation_context: Contexte de conversation optionnel
            
        Returns:
            Classification d'intention
        """
        # Préparer le prompt pour la classification
        system_prompt = get_intent_classification_prompt()
        
        # Ajouter le contexte de conversation si disponible
        context_text = ""
        if conversation_context and len(conversation_context) > 1:
            # Prendre les 3 derniers messages pour le contexte
            recent_context = conversation_context[-3:]
            context_text = "Contexte de la conversation:\n" + "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in recent_context
            ])
        
        # Construire le message utilisateur
        user_prompt = f"Classifiez cette requête utilisateur:\n\n{query}"
        if context_text:
            user_prompt += f"\n\n{context_text}"
        
        # Appeler le LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.llm_service.generate_response(
            messages=messages,
            temperature=0.3,  # Température basse pour des résultats plus déterministes
            max_tokens=1000,
            stream=False
        )
        
        # Extraire la classification JSON
        try:
            # Nettoyer la réponse et extraire le JSON
            json_text = response
            
            # Supprimer les blocs de code markdown s'ils existent
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            # Extraire le premier objet JSON trouvé
            json_pattern = r'\{.*\}'
            json_matches = re.search(json_pattern, json_text, re.DOTALL)
            if json_matches:
                json_text = json_matches.group(0)
            
            # Analyser le JSON
            classification_data = json.loads(json_text)
            
            # Extraire et normaliser les données
            intent_str = classification_data.get("intent", "GENERAL_QUERY")
            confidence = float(classification_data.get("confidence", 0.7))
            entities = classification_data.get("entities", {})
            
            # Convertir en type d'intention
            try:
                intent_type = IntentType(intent_str)
            except ValueError:
                # Si l'intention n'est pas reconnue, utiliser GENERAL_QUERY
                intent_type = IntentType.GENERAL_QUERY
                confidence *= 0.8  # Réduire la confiance
            
            return IntentClassification(
                intent=intent_type,
                confidence=confidence,
                entities=entities,
                raw_response=response
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Erreur lors de l'analyse de la réponse LLM: {str(e)}")
            logger.debug(f"Réponse LLM brute: {response}")
            
            # En cas d'erreur, retourner une classification par défaut
            return IntentClassification(
                intent=IntentType.GENERAL_QUERY,
                confidence=0.5,
                entities={},
                raw_response=response
            )
    
    def _extract_basic_entities(self, query: str) -> Dict[str, Any]:
        """
        Extrait des entités basiques d'une requête.
        
        Args:
            query: Requête à analyser
            
        Returns:
            Dictionnaire des entités extraites
        """
        entities = {}
        
        # Extraire les dates (format JJ/MM/AAAA ou AAAA-MM-JJ)
        date_patterns = [
            r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2,4})',  # JJ/MM/AAAA ou variations
            r'(\d{4})[/\.-](\d{1,2})[/\.-](\d{1,2})'     # AAAA-MM-JJ ou variations
        ]
        
        # Mots-clés temporels
        time_keywords = {
            "aujourd'hui": datetime.now().date(),
            "hier": (datetime.now().date().replace(day=datetime.now().date().day - 1)),
            "ce mois": (datetime.now().date().replace(day=1)),
            "mois dernier": (datetime.now().date().replace(month=datetime.now().date().month - 1, day=1)),
            "cette année": (datetime.now().date().replace(month=1, day=1)),
            "l'année dernière": (datetime.now().date().replace(year=datetime.now().date().year - 1, month=1, day=1))
        }
        
        for keyword, date_value in time_keywords.items():
            if keyword.lower() in query.lower():
                if "date_start" not in entities:
                    entities["date_start"] = date_value.isoformat()
        
        # Extraire les montants (ex: 100€, 100 euros, 100.50)
        amount_patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:€|euros?|EUR)',  # 100€, 100 euros
            r'(\d+(?:[.,]\d+)?)'                      # Nombre simple (avec heuristique)
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                # Convertir la virgule en point pour le float
                amount_str = matches[0].replace(',', '.')
                amount = float(amount_str)
                
                # Déterminer s'il s'agit d'un min_amount ou max_amount selon le contexte
                if re.search(r'plus\s+(?:de|que)|supérieur|minimum', query, re.IGNORECASE):
                    entities["min_amount"] = amount
                elif re.search(r'moins\s+(?:de|que)|inférieur|maximum', query, re.IGNORECASE):
                    entities["max_amount"] = amount
                else:
                    entities["amount"] = amount
                
                break
        
        # Extraire les catégories et commerçants (basique, à améliorer avec NER)
        category_matches = re.findall(r'catégorie\s+(\w+)', query, re.IGNORECASE)
        if category_matches:
            entities["category"] = category_matches[0]
        
        merchant_matches = re.findall(r'(?:chez|à|au|aux)\s+(\w+)', query, re.IGNORECASE)
        if merchant_matches:
            entities["merchant"] = merchant_matches[0]
        
        return entities