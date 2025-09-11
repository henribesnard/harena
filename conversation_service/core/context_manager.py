"""
Context Manager - Agent Logique Phase 3
Architecture v2.0 - Composant déterministe

Responsabilité : Compression intelligente des tokens
- Gestion contexte conversationnel
- Compression tokens pour optimisation LLM
- Historique conversation avec TTL
- Résumé intelligent des échanges précédents
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Un tour de conversation (user message + assistant response)"""
    user_message: str
    assistant_response: str
    intent_detected: Optional[str]
    entities_extracted: List[Dict[str, Any]]
    timestamp: datetime
    processing_time_ms: int
    token_count: int = 0

@dataclass 
class ContextSnapshot:
    """Snapshot du contexte à un moment donné"""
    conversation_id: str
    user_id: int
    turns_count: int
    total_tokens: int
    compressed_tokens: int
    compression_ratio: float
    created_at: datetime
    summary: Optional[str] = None

@dataclass
class ContextCompressionRequest:
    """Requête de compression de contexte"""
    conversation_id: str
    max_tokens: int = 4000
    preserve_last_turns: int = 3
    compression_strategy: str = "summarize"  # summarize, truncate, intelligent

@dataclass
class ContextCompressionResult:
    """Résultat de compression de contexte"""
    success: bool
    compressed_context: List[ConversationTurn]
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    summary_generated: Optional[str] = None
    processing_time_ms: int = 0

class ContextManager:
    """
    Agent logique pour gestion du contexte conversationnel
    
    Optimise la mémoire et les tokens pour les appels LLM
    """
    
    def __init__(
        self,
        max_context_turns: int = 10,
        max_total_tokens: int = 8000,
        context_ttl_hours: int = 24,
        enable_compression: bool = True
    ):
        self.max_context_turns = max_context_turns
        self.max_total_tokens = max_total_tokens
        self.context_ttl_hours = context_ttl_hours
        self.enable_compression = enable_compression
        
        # Stockage des conversations actives
        self._conversations: Dict[str, deque] = {}
        self._conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Cache des résumés
        self._summary_cache: Dict[str, str] = {}
        
        # Statistiques
        self.stats = {
            "conversations_active": 0,
            "total_turns_stored": 0,
            "compressions_performed": 0,
            "tokens_saved": 0,
            "cache_hits": 0
        }
        
        logger.info("ContextManager initialisé")
    
    async def add_conversation_turn(
        self,
        conversation_id: str,
        user_id: int,
        user_message: str,
        assistant_response: str,
        intent_detected: Optional[str] = None,
        entities_extracted: List[Dict[str, Any]] = None,
        processing_time_ms: int = 0
    ) -> bool:
        """
        Ajoute un nouveau tour de conversation
        
        Args:
            conversation_id: ID de la conversation
            user_id: ID de l'utilisateur
            user_message: Message utilisateur
            assistant_response: Réponse assistant
            intent_detected: Intention détectée
            entities_extracted: Entités extraites
            processing_time_ms: Temps de traitement
            
        Returns:
            bool: Succès de l'ajout
        """
        try:
            # Initialisation conversation si nécessaire
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = deque(maxlen=self.max_context_turns)
                self._conversation_metadata[conversation_id] = {
                    "user_id": user_id,
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "turn_count": 0,
                    "total_tokens": 0
                }
                self.stats["conversations_active"] += 1
            
            # Estimation tokens (approximation)
            token_count = self._estimate_tokens(user_message + assistant_response)
            
            # Création du tour
            turn = ConversationTurn(
                user_message=user_message,
                assistant_response=assistant_response,
                intent_detected=intent_detected,
                entities_extracted=entities_extracted or [],
                timestamp=datetime.now(),
                processing_time_ms=processing_time_ms,
                token_count=token_count
            )
            
            # Ajout à la conversation
            self._conversations[conversation_id].append(turn)
            
            # Mise à jour métadonnées
            metadata = self._conversation_metadata[conversation_id]
            metadata["last_activity"] = datetime.now()
            metadata["turn_count"] += 1
            metadata["total_tokens"] += token_count
            
            self.stats["total_turns_stored"] += 1
            
            # Auto-compression si nécessaire
            if (self.enable_compression and 
                metadata["total_tokens"] > self.max_total_tokens):
                await self._auto_compress_conversation(conversation_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur ajout tour conversation {conversation_id}: {str(e)}")
            return False
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        max_turns: Optional[int] = None,
        include_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Récupère le contexte conversationnel optimisé
        
        Args:
            conversation_id: ID de la conversation
            max_turns: Nombre max de tours à récupérer
            include_summary: Inclure résumé si disponible
            
        Returns:
            List[Dict]: Contexte conversationnel formaté
        """
        try:
            if conversation_id not in self._conversations:
                return []
            
            turns = list(self._conversations[conversation_id])
            
            # Limitation nombre de tours
            if max_turns:
                turns = turns[-max_turns:]
            
            # Construction contexte
            context = []
            
            # Ajout résumé si demandé et disponible
            if include_summary and conversation_id in self._summary_cache:
                context.append({
                    "type": "summary",
                    "content": self._summary_cache[conversation_id],
                    "timestamp": None
                })
                self.stats["cache_hits"] += 1
            
            # Ajout tours de conversation
            for turn in turns:
                context.extend([
                    {
                        "type": "user_message",
                        "content": turn.user_message,
                        "timestamp": turn.timestamp.isoformat(),
                        "intent": turn.intent_detected,
                        "entities": turn.entities_extracted
                    },
                    {
                        "type": "assistant_response", 
                        "content": turn.assistant_response,
                        "timestamp": turn.timestamp.isoformat(),
                        "processing_time_ms": turn.processing_time_ms
                    }
                ])
            
            return context
            
        except Exception as e:
            logger.error(f"Erreur récupération contexte {conversation_id}: {str(e)}")
            return []
    
    async def compress_conversation_context(
        self,
        request: ContextCompressionRequest
    ) -> ContextCompressionResult:
        """
        Compresse le contexte conversationnel selon la stratégie demandée
        
        Args:
            request: Paramètres de compression
            
        Returns:
            ContextCompressionResult: Résultat compression
        """
        start_time = datetime.now()
        
        try:
            conversation_id = request.conversation_id
            
            if conversation_id not in self._conversations:
                return ContextCompressionResult(
                    success=False,
                    compressed_context=[],
                    original_token_count=0,
                    compressed_token_count=0,
                    compression_ratio=0.0,
                    processing_time_ms=self._get_processing_time(start_time)
                )
            
            turns = list(self._conversations[conversation_id])
            original_token_count = sum(turn.token_count for turn in turns)
            
            # Application stratégie de compression
            if request.compression_strategy == "truncate":
                compressed_turns = await self._truncate_compression(turns, request)
                summary = None
                
            elif request.compression_strategy == "summarize":
                compressed_turns, summary = await self._summarize_compression(turns, request)
                
            elif request.compression_strategy == "intelligent":
                compressed_turns, summary = await self._intelligent_compression(turns, request)
                
            else:
                # Fallback: truncate
                compressed_turns = await self._truncate_compression(turns, request)
                summary = None
            
            compressed_token_count = sum(turn.token_count for turn in compressed_turns)
            
            # Calcul ratio compression
            compression_ratio = 0.0
            if original_token_count > 0:
                compression_ratio = 1.0 - (compressed_token_count / original_token_count)
            
            # Mise à jour conversation compressée
            self._conversations[conversation_id] = deque(compressed_turns, maxlen=self.max_context_turns)
            
            # Cache du résumé si généré
            if summary:
                self._summary_cache[conversation_id] = summary
            
            # Statistiques
            self.stats["compressions_performed"] += 1
            self.stats["tokens_saved"] += (original_token_count - compressed_token_count)
            
            return ContextCompressionResult(
                success=True,
                compressed_context=compressed_turns,
                original_token_count=original_token_count,
                compressed_token_count=compressed_token_count,
                compression_ratio=compression_ratio,
                summary_generated=summary,
                processing_time_ms=self._get_processing_time(start_time)
            )
            
        except Exception as e:
            logger.error(f"Erreur compression contexte {request.conversation_id}: {str(e)}")
            return ContextCompressionResult(
                success=False,
                compressed_context=[],
                original_token_count=0,
                compressed_token_count=0,
                compression_ratio=0.0,
                processing_time_ms=self._get_processing_time(start_time)
            )
    
    async def _auto_compress_conversation(self, conversation_id: str) -> None:
        """Compression automatique d'une conversation"""
        
        request = ContextCompressionRequest(
            conversation_id=conversation_id,
            max_tokens=self.max_total_tokens // 2,  # Cible 50% du max
            preserve_last_turns=3,
            compression_strategy="intelligent"
        )
        
        result = await self.compress_conversation_context(request)
        
        if result.success:
            logger.info(f"Auto-compression {conversation_id}: "
                       f"{result.compression_ratio:.1%} tokens saved")
        else:
            logger.warning(f"Échec auto-compression {conversation_id}")
    
    async def _truncate_compression(
        self, 
        turns: List[ConversationTurn], 
        request: ContextCompressionRequest
    ) -> List[ConversationTurn]:
        """Compression par troncature - garde les derniers tours"""
        
        # Garde les N derniers tours selon preserve_last_turns
        preserved_turns = turns[-request.preserve_last_turns:] if turns else []
        
        # Vérifie la limite de tokens
        total_tokens = sum(turn.token_count for turn in preserved_turns)
        
        # Supprime progressivement les plus anciens si dépassement
        while preserved_turns and total_tokens > request.max_tokens:
            removed_turn = preserved_turns.pop(0)
            total_tokens -= removed_turn.token_count
        
        return preserved_turns
    
    async def _summarize_compression(
        self, 
        turns: List[ConversationTurn], 
        request: ContextCompressionRequest
    ) -> Tuple[List[ConversationTurn], str]:
        """Compression par résumé - génère un résumé des anciens tours"""
        
        if len(turns) <= request.preserve_last_turns:
            return turns, ""
        
        # Tours à résumer (tous sauf les derniers à préserver)
        turns_to_summarize = turns[:-request.preserve_last_turns]
        turns_to_keep = turns[-request.preserve_last_turns:]
        
        # Génération résumé simple (peut être amélioré avec LLM)
        summary = await self._generate_simple_summary(turns_to_summarize)
        
        return turns_to_keep, summary
    
    async def _intelligent_compression(
        self, 
        turns: List[ConversationTurn], 
        request: ContextCompressionRequest
    ) -> Tuple[List[ConversationTurn], str]:
        """Compression intelligente - combine résumé et sélection pertinente"""
        
        # Stratégie hybride:
        # 1. Garde les derniers tours (plus récents)
        # 2. Sélectionne quelques tours importants du milieu
        # 3. Résume le reste
        
        if len(turns) <= request.preserve_last_turns:
            return turns, ""
        
        # Derniers tours (toujours gardés)
        recent_turns = turns[-request.preserve_last_turns:]
        
        # Tours plus anciens à analyser
        older_turns = turns[:-request.preserve_last_turns]
        
        if not older_turns:
            return recent_turns, ""
        
        # Sélection intelligente des tours importants
        important_turns = self._select_important_turns(older_turns, max_turns=2)
        
        # Résumé des tours non sélectionnés
        turns_to_summarize = [turn for turn in older_turns if turn not in important_turns]
        summary = await self._generate_simple_summary(turns_to_summarize) if turns_to_summarize else ""
        
        # Reconstitution ordre chronologique
        final_turns = important_turns + recent_turns
        final_turns.sort(key=lambda t: t.timestamp)
        
        return final_turns, summary
    
    def _select_important_turns(
        self, 
        turns: List[ConversationTurn], 
        max_turns: int = 2
    ) -> List[ConversationTurn]:
        """Sélectionne les tours les plus importants selon critères heuristiques"""
        
        # Scoring des tours selon différents critères
        scored_turns = []
        
        for turn in turns:
            score = 0.0
            
            # Bonus si intention détectée
            if turn.intent_detected and turn.intent_detected != "UNCLEAR_INTENT":
                score += 2.0
            
            # Bonus si entités extraites
            if turn.entities_extracted:
                score += len(turn.entities_extracted) * 0.5
            
            # Bonus pour longueur raisonnable (ni trop court ni trop long)
            msg_length = len(turn.user_message)
            if 20 <= msg_length <= 200:
                score += 1.0
            
            # Malus pour tours très récents (seront déjà gardés)
            minutes_ago = (datetime.now() - turn.timestamp).total_seconds() / 60
            if minutes_ago < 10:
                score -= 1.0
            
            scored_turns.append((turn, score))
        
        # Tri par score décroissant
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        
        # Sélection des meilleurs
        return [turn for turn, score in scored_turns[:max_turns]]
    
    async def _generate_simple_summary(
        self, 
        turns: List[ConversationTurn]
    ) -> str:
        """Génère un résumé simple des tours de conversation"""
        
        if not turns:
            return ""
        
        # Extraction des informations clés
        intents = [turn.intent_detected for turn in turns if turn.intent_detected]
        entities = []
        for turn in turns:
            entities.extend(turn.entities_extracted or [])
        
        # Construction résumé basique
        summary_parts = []
        
        if len(turns) == 1:
            summary_parts.append("L'utilisateur a fait 1 demande précédente.")
        else:
            summary_parts.append(f"L'utilisateur a fait {len(turns)} demandes précédentes.")
        
        # Résumé des intentions
        if intents:
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            most_frequent = max(intent_counts.items(), key=lambda x: x[1])
            summary_parts.append(f"Principalement des demandes de type {most_frequent[0]}.")
        
        # Résumé des entités
        if entities:
            entity_types = {entity.get("type", "unknown") for entity in entities}
            summary_parts.append(f"Entités mentionnées: {', '.join(entity_types)}.")
        
        return " ".join(summary_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimation approximative du nombre de tokens"""
        
        # Approximation basique: 1 token  4 caractères pour l'anglais/français
        # Plus sophistiqué avec un tokenizer réel en production
        return max(1, len(text) // 4)
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calcule le temps de traitement en ms"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    async def cleanup_expired_conversations(self) -> int:
        """Nettoie les conversations expirées selon TTL"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.context_ttl_hours)
        expired_conversations = []
        
        for conversation_id, metadata in self._conversation_metadata.items():
            if metadata["last_activity"] < cutoff_time:
                expired_conversations.append(conversation_id)
        
        # Suppression des conversations expirées
        for conversation_id in expired_conversations:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
            if conversation_id in self._conversation_metadata:
                del self._conversation_metadata[conversation_id]
            if conversation_id in self._summary_cache:
                del self._summary_cache[conversation_id]
        
        self.stats["conversations_active"] -= len(expired_conversations)
        
        if expired_conversations:
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
        
        return len(expired_conversations)
    
    def get_conversation_snapshot(self, conversation_id: str) -> Optional[ContextSnapshot]:
        """Récupère un snapshot d'une conversation"""
        
        if conversation_id not in self._conversations:
            return None
        
        turns = list(self._conversations[conversation_id])
        metadata = self._conversation_metadata.get(conversation_id, {})
        
        total_tokens = sum(turn.token_count for turn in turns)
        
        return ContextSnapshot(
            conversation_id=conversation_id,
            user_id=metadata.get("user_id", 0),
            turns_count=len(turns),
            total_tokens=total_tokens,
            compressed_tokens=total_tokens,  # Si pas de compression active
            compression_ratio=0.0,
            created_at=metadata.get("created_at", datetime.now()),
            summary=self._summary_cache.get(conversation_id)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques du ContextManager"""
        
        # Calculs statistiques additionnels
        total_active_tokens = sum(
            sum(turn.token_count for turn in turns)
            for turns in self._conversations.values()
        )
        
        avg_turns_per_conversation = 0.0
        if self.stats["conversations_active"] > 0:
            avg_turns_per_conversation = (
                self.stats["total_turns_stored"] / self.stats["conversations_active"]
            )
        
        return {
            **self.stats,
            "total_active_tokens": total_active_tokens,
            "avg_turns_per_conversation": avg_turns_per_conversation,
            "cache_size": len(self._summary_cache)
        }
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Supprime complètement une conversation"""
        
        try:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
            if conversation_id in self._conversation_metadata:
                del self._conversation_metadata[conversation_id]
            if conversation_id in self._summary_cache:
                del self._summary_cache[conversation_id]
            
            self.stats["conversations_active"] = max(0, self.stats["conversations_active"] - 1)
            
            logger.info(f"Conversation {conversation_id} supprimée")
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression conversation {conversation_id}: {str(e)}")
            return False