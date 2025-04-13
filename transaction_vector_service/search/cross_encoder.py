# transaction_vector_service/search/cross_encoder.py
"""
Module de reranking avec cross-encoder pour les transactions.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from ..config.logging_config import get_logger
from ..utils.text_processors import clean_transaction_description

logger = get_logger(__name__)

class CrossEncoderRanker:
    """
    Service de reranking utilisant un cross-encoder pour les transactions.
    """

    def __init__(self):
        """
        Initialise le service de cross-encoder.
        """
        # Dans une implémentation réelle, on initialiserait ici le modèle cross-encoder
        # Par exemple, avec SentenceTransformers
        # Exemple : self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Par souci de simplicité, cette implémentation utilise une fonction de score simple
        logger.info("Service de cross-encoder initialisé")

    async def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Réordonne les résultats de recherche en utilisant un cross-encoder.
        
        Args:
            query: Requête de recherche
            candidates: Candidats à réordonner
            
        Returns:
            Liste réordonnée de résultats
        """
        if not candidates:
            return []
            
        logger.info(f"Reranking de {len(candidates)} résultats pour la requête '{query}'")
        
        # Préparer les paires (requête, document) pour le cross-encoder
        pairs = []
        for candidate in candidates:
            # Extraire le texte du document
            doc_text = self._extract_document_text(candidate)
            pairs.append((query, doc_text))
        
        # Dans une implémentation réelle, on utiliserait le modèle cross-encoder
        # pour calculer les scores pour toutes les paires
        # Exemple : scores = self.model.predict(pairs)
        
        # Simulation de scores cross-encoder basée sur les scores initiaux et la longueur du texte
        cross_encoder_scores = await self._simulate_cross_encoder_scores(query, candidates, pairs)
        
        # Combiner les scores avec la pondération
        for i, candidate in enumerate(candidates):
            cross_score = cross_encoder_scores[i]
            bm25_score = candidate.get("bm25_score", 0)
            vector_score = candidate.get("vector_score", 0)
            initial_score = candidate.get("combined_initial_score", 0)
            
            # Score final pondéré
            final_score = (
                initial_score * 0.6 +  # Score initial déjà pondéré (BM25 + vectoriel)
                cross_score * 0.4      # Score cross-encoder
            )
            
            candidate["cross_encoder_score"] = cross_score
            candidate["final_score"] = final_score
            
        # Trier par score final
        reranked = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)
        
        logger.info("Reranking terminé")
        return reranked

    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """
        Extrait le texte pertinent d'un document transaction.
        
        Args:
            document: Document transaction
            
        Returns:
            Texte du document
        """
        texts = []
        if "clean_description" in document and document["clean_description"]:
            texts.append(document["clean_description"])
        elif "description" in document and document["description"]:
            texts.append(document["description"])
            
        if "normalized_merchant" in document and document["normalized_merchant"]:
            texts.append(document["normalized_merchant"])
            
        if "category_name" in document and document["category_name"]:
            texts.append(document["category_name"])
            
        # Ajouter le montant comme texte
        if "amount" in document:
            texts.append(f"montant: {document['amount']}")
            
        return " ".join(texts)

    async def _simulate_cross_encoder_scores(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Simule des scores de cross-encoder.
        Dans une implémentation réelle, on utiliserait un vrai modèle.
        
        Args:
            query: Requête
            candidates: Candidats
            pairs: Paires (requête, document)
            
        Returns:
            Liste de scores
        """
        # Cette fonction simule les scores d'un cross-encoder
        # Dans une implémentation réelle, ce serait le résultat d'un modèle de ML
        
        scores = []
        clean_query = clean_transaction_description(query.lower())
        query_tokens = set(clean_query.split())
        
        for i, (candidate, pair) in enumerate(zip(candidates, pairs)):
            # Facteurs influençant le score
            # 1. Présence des termes de la requête dans le document
            doc_text = pair[1].lower()
            term_overlap = sum(1 for token in query_tokens if token in doc_text) / max(1, len(query_tokens))
            
            # 2. Scores existants (BM25 et vectoriel)
            bm25_score = candidate.get("bm25_score", 0)
            vector_score = candidate.get("vector_score", 0)
            
            # 3. Facteur de récence (transactions plus récentes = plus pertinentes)
            recency = 0.0
            if "transaction_date" in candidate:
                # Simple heuristique: plus c'est récent, plus le score est élevé
                # Dans une vraie implémentation, ce serait calculé par rapport à la date actuelle
                recency = 0.2  # Valeur par défaut modérée
            
            # Combiner les facteurs pour simuler un score cross-encoder
            # La formule est une approximation simple de ce qu'un cross-encoder pourrait produire
            cross_score = (
                term_overlap * 0.4 +
                (bm25_score / 10) * 0.3 +  # Normalisé car BM25 peut donner des scores élevés
                vector_score * 0.2 +
                recency * 0.1
            )
            
            # Ajouter un peu de bruit pour simuler la variabilité d'un modèle réel
            noise = np.random.normal(0, 0.05)
            cross_score = max(0, min(1, cross_score + noise))
            
            scores.append(cross_score)
            
        return scores