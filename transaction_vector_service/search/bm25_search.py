# transaction_vector_service/search/bm25_search.py
"""
Module de recherche lexicale BM25 pour les transactions.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio

from ..config.logging_config import get_logger
from ..models.transaction import TransactionSearch
from ..utils.text_processors import clean_transaction_description
from ..services.transaction_service import TransactionService

logger = get_logger(__name__)

class BM25Search:
    """
    Service de recherche lexicale utilisant l'algorithme BM25 pour les transactions.
    """

    def __init__(
        self,
        transaction_service: Optional[TransactionService] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialise le service de recherche BM25.
        
        Args:
            transaction_service: Service de transaction
            k1: Paramètre k1 de BM25 (saturation de fréquence des termes)
            b: Paramètre b de BM25 (normalisation de longueur)
        """
        self.transaction_service = transaction_service or TransactionService()
        self.k1 = k1
        self.b = b
        
        # Indexation en mémoire
        self._index = {}  # Terme -> {doc_id -> fréquence}
        self._doc_lengths = {}  # doc_id -> longueur du document
        self._avg_doc_length = 0
        self._total_docs = 0
        self._indexed_users = set()  # Ensemble des utilisateurs indexés
        
        logger.info("Service de recherche BM25 initialisé")

    async def ensure_user_indexed(self, user_id: int) -> None:
        """
        S'assure que les transactions de l'utilisateur sont indexées.
        
        Args:
            user_id: ID de l'utilisateur
        """
        if user_id in self._indexed_users:
            return
            
        logger.info(f"Indexation des transactions pour l'utilisateur {user_id}")
        
        # Récupérer toutes les transactions de l'utilisateur
        transactions, _ = await self.transaction_service.search_transactions(
            user_id=user_id,
            search_params=TransactionSearch(
                limit=10000,  # Limiter à un nombre raisonnable
                include_future=False,
                include_deleted=False
            )
        )
        
        # Indexer les transactions
        for tx in transactions:
            self._index_transaction(tx)
            
        # Mettre à jour les statistiques globales
        self._total_docs = len(self._doc_lengths)
        if self._total_docs > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._total_docs
            
        # Marquer l'utilisateur comme indexé
        self._indexed_users.add(user_id)
        
        logger.info(f"Indexation terminée: {len(transactions)} transactions pour l'utilisateur {user_id}")

    def _index_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Indexe une transaction dans l'index BM25.
        
        Args:
            transaction: Transaction à indexer
        """
        doc_id = transaction.get("id")
        if not doc_id:
            return
            
        # Récupérer le texte à indexer
        texts = []
        if "description" in transaction:
            texts.append(transaction["description"])
        if "clean_description" in transaction:
            texts.append(transaction["clean_description"])
        if "normalized_merchant" in transaction:
            texts.append(transaction["normalized_merchant"])
            
        # Combiner et nettoyer le texte
        text = " ".join(filter(None, texts))
        clean_text = clean_transaction_description(text)
        
        # Tokenisation simple
        tokens = re.findall(r'\b\w+\b', clean_text.lower())
        
        # Compter les occurrences des termes
        term_counts = {}
        for token in tokens:
            if len(token) < 2:  # Ignorer les termes trop courts
                continue
                
            if token in term_counts:
                term_counts[token] += 1
            else:
                term_counts[token] = 1
        
        # Mettre à jour l'index
        for term, count in term_counts.items():
            if term not in self._index:
                self._index[term] = {}
            self._index[term][doc_id] = count
        
        # Stocker la longueur du document (nombre de termes)
        self._doc_lengths[doc_id] = len(tokens)

    async def search(
        self, 
        user_id: int,
        query: str,
        search_params: TransactionSearch,
        top_k: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Exécute une recherche BM25 sur les transactions.
        
        Args:
            user_id: ID de l'utilisateur
            query: Requête de recherche
            search_params: Paramètres de recherche supplémentaires
            top_k: Nombre maximum de résultats
            
        Returns:
            Tuple de (résultats de recherche, nombre total)
        """
        # S'assurer que les transactions de l'utilisateur sont indexées
        await self.ensure_user_indexed(user_id)
        
        # Nettoyer la requête
        clean_query = clean_transaction_description(query)
        
        # Tokeniser la requête
        query_terms = re.findall(r'\b\w+\b', clean_query.lower())
        query_terms = [t for t in query_terms if len(t) >= 2]  # Ignorer les termes trop courts
        
        if not query_terms:
            logger.info(f"Aucun terme de recherche valide dans la requête '{query}'")
            return [], 0
            
        # Calculer les scores BM25
        scores = {}
        for term in query_terms:
            if term not in self._index:
                continue
                
            # Nombre de documents contenant ce terme
            df = len(self._index[term])
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # Pour chaque document contenant ce terme
            for doc_id, term_freq in self._index[term].items():
                # Récupérer la longueur du document
                doc_length = self._doc_lengths.get(doc_id, self._avg_doc_length)
                
                # Calculer le score BM25 pour ce terme dans ce document
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
                term_score = idf * numerator / denominator
                
                # Ajouter au score total du document
                if doc_id in scores:
                    scores[doc_id] += term_score
                else:
                    scores[doc_id] = term_score
        
        # Trier les documents par score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Récupérer les documents complets
        results = []
        for doc_id, score in sorted_scores[:top_k]:
            # Récupérer la transaction complète
            tx = await self.transaction_service.get_transaction(doc_id)
            if tx:
                tx["score"] = score
                results.append(tx)
        
        logger.info(f"Recherche BM25 terminée: {len(results)} résultats pour '{query}'")
        return results, len(sorted_scores)