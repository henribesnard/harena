"""
Moteur de recherche unifié pour le service de recherche Harena.

Ce module fournit une interface unifiée pour interagir avec les différents
moteurs de recherche (BM25, Whoosh, Qdrant) pour simplifier l'intégration.
"""
import logging
import os
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import time

# Imports conditionnels des différents moteurs
from search_service.storage.bm25_engine import get_bm25f_engine, BM25FEngine
from search_service.storage.whoosh_engine import get_whoosh_engine, WhooshSearchEngine

# Import conditionnel de Qdrant pour la recherche vectorielle
try:
    from search_service.storage.qdrant import get_qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from config_service.config import settings

logger = logging.getLogger(__name__)

class SearchEngineType(str, Enum):
    """Types de moteurs de recherche disponibles."""
    BM25 = "bm25"
    WHOOSH = "whoosh"
    UNIFIED = "unified"

class UnifiedSearchEngine:
    """
    Moteur de recherche unifié qui orchestre les différents moteurs.
    """
    
    def __init__(self, primary_engine: SearchEngineType = None):
        """
        Initialise le moteur de recherche unifié.
        
        Args:
            primary_engine: Type de moteur principal à utiliser
        """
        # Déterminer le moteur principal à utiliser
        self.primary_engine_type = primary_engine or SearchEngineType(
            os.environ.get("PRIMARY_SEARCH_ENGINE", SearchEngineType.WHOOSH)
        )
        
        logger.info(f"Initialisation du moteur de recherche unifié avec {self.primary_engine_type} comme moteur principal")
        
        # Initialiser les moteurs disponibles
        self.bm25_engine = get_bm25f_engine()
        self.whoosh_engine = get_whoosh_engine()
        
        # Statistiques
        self.engine_usage = {
            SearchEngineType.BM25: 0,
            SearchEngineType.WHOOSH: 0,
            SearchEngineType.UNIFIED: 0
        }
        
        # Indicateur d'initialisation des modèles vectoriels
        self.vector_initialized = QDRANT_AVAILABLE
    
    async def index_documents(self, user_id: int, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexe un lot de documents dans tous les moteurs de recherche.
        
        Args:
            user_id: ID de l'utilisateur
            documents: Liste de documents à indexer
            
        Returns:
            Résultat de l'indexation
        """
        start_time = time.time()
        results = {"status": "pending", "engines": {}}
        
        # Indexer dans BM25
        try:
            bm25_success = self.bm25_engine.index_documents(user_id, documents)
            results["engines"][SearchEngineType.BM25] = {
                "status": "success" if bm25_success else "error",
                "documents_processed": len(documents)
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation BM25 pour l'utilisateur {user_id}: {e}", exc_info=True)
            results["engines"][SearchEngineType.BM25] = {
                "status": "error",
                "message": str(e)
            }
        
        # Indexer dans Whoosh
        try:
            whoosh_success = self.whoosh_engine.index_documents(user_id, documents)
            results["engines"][SearchEngineType.WHOOSH] = {
                "status": "success" if whoosh_success else "error",
                "documents_processed": len(documents)
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation Whoosh pour l'utilisateur {user_id}: {e}", exc_info=True)
            results["engines"][SearchEngineType.WHOOSH] = {
                "status": "error",
                "message": str(e)
            }
        
        # Indexer dans Qdrant pour la recherche vectorielle (si disponible)
        if QDRANT_AVAILABLE:
            try:
                from sync_service.services.vector_storage import VectorStorageService
                vector_storage = VectorStorageService()
                
                # Préparer les documents pour Qdrant (format spécifique)
                qdrant_docs = []
                for doc in documents:
                    # Adapter au format attendu par la fonction batch_store_transactions
                    qdrant_doc = {
                        "user_id": user_id,
                        "bridge_transaction_id": doc.get("bridge_transaction_id", ""),
                        "account_id": doc.get("account_id"),
                        "amount": doc.get("amount", 0.0),
                        "currency_code": doc.get("currency_code", ""),
                        "description": doc.get("description", ""),
                        "clean_description": doc.get("clean_description", ""),
                        "transaction_date": doc.get("transaction_date"),
                        "category_id": doc.get("category_id"),
                        "operation_type": doc.get("operation_type"),
                        "merchant_id": doc.get("merchant_id"),
                    }
                    qdrant_docs.append(qdrant_doc)
                
                vector_result = await vector_storage.batch_store_transactions(qdrant_docs)
                results["engines"]["qdrant"] = vector_result
            except Exception as e:
                logger.error(f"Erreur lors de l'indexation Qdrant pour l'utilisateur {user_id}: {e}", exc_info=True)
                results["engines"]["qdrant"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Déterminer le statut global
        all_success = all(engine["status"] == "success" for engine in results["engines"].values())
        any_success = any(engine["status"] == "success" for engine in results["engines"].values())
        
        if all_success:
            results["status"] = "success"
        elif any_success:
            results["status"] = "partial"
        else:
            results["status"] = "error"
        
        # Ajouter les statistiques d'indexation
        elapsed_time = time.time() - start_time
        results["execution_time_ms"] = int(elapsed_time * 1000)
        results["documents_count"] = len(documents)
        
        logger.info(f"Indexation unifiée terminée pour l'utilisateur {user_id}: {results['status']} ({len(documents)} documents en {elapsed_time:.2f}s)")
        return results
    
    async def search(self, 
                    user_id: int, 
                    query_text: str, 
                    engine_type: Optional[SearchEngineType] = None,
                    field_weights: Optional[Dict[str, float]] = None,
                    top_k: int = 50,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Effectue une recherche dans les documents d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query_text: Texte de la requête
            engine_type: Type de moteur à utiliser (utilise le moteur principal par défaut)
            field_weights: Poids des champs (optionnel)
            top_k: Nombre de résultats à retourner
            filters: Filtres additionnels (optionnel)
            
        Returns:
            Liste des résultats de recherche
        """
        start_time = time.time()
        
        # Utiliser le moteur principal si non spécifié
        engine_type = engine_type or self.primary_engine_type
        
        # Incrémenter le compteur d'utilisation
        self.engine_usage[engine_type] += 1
        
        try:
            # Recherche avec le moteur spécifié
            if engine_type == SearchEngineType.BM25:
                results = self.bm25_engine.search(
                    user_id=user_id,
                    query_text=query_text,
                    field_weights=field_weights,
                    top_k=top_k
                )
            elif engine_type == SearchEngineType.WHOOSH:
                results = self.whoosh_engine.search(
                    user_id=user_id,
                    query_text=query_text,
                    field_weights=field_weights,
                    top_k=top_k,
                    filters=filters
                )
            elif engine_type == SearchEngineType.UNIFIED:
                # Mode unifié: combiner les résultats de tous les moteurs
                # Pour cet exemple, on utilise simplement les résultats du moteur principal
                # Une implémentation plus sophistiquée combinerait les résultats de plusieurs moteurs
                results = self.whoosh_engine.search(
                    user_id=user_id,
                    query_text=query_text,
                    field_weights=field_weights,
                    top_k=top_k,
                    filters=filters
                )
            else:
                logger.error(f"Type de moteur non reconnu: {engine_type}")
                return []
            
            elapsed_time = time.time() - start_time
            logger.info(f"Recherche '{query_text}' avec {engine_type}: {len(results)} résultats en {elapsed_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche pour l'utilisateur {user_id} avec {engine_type}: {e}", exc_info=True)
            return []
    
    async def delete_user_index(self, user_id: int) -> Dict[str, Any]:
        """
        Supprime l'index d'un utilisateur de tous les moteurs.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Résultat de la suppression
        """
        results = {"status": "pending", "engines": {}}
        
        # Supprimer de BM25
        try:
            bm25_success = self.bm25_engine.delete_user_index(user_id)
            results["engines"][SearchEngineType.BM25] = {
                "status": "success" if bm25_success else "error"
            }
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'index BM25 pour l'utilisateur {user_id}: {e}", exc_info=True)
            results["engines"][SearchEngineType.BM25] = {
                "status": "error",
                "message": str(e)
            }
        
        # Supprimer de Whoosh
        try:
            whoosh_success = self.whoosh_engine.delete_user_index(user_id)
            results["engines"][SearchEngineType.WHOOSH] = {
                "status": "success" if whoosh_success else "error"
            }
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'index Whoosh pour l'utilisateur {user_id}: {e}", exc_info=True)
            results["engines"][SearchEngineType.WHOOSH] = {
                "status": "error",
                "message": str(e)
            }
        
        # Supprimer de Qdrant (si disponible)
        if QDRANT_AVAILABLE:
            try:
                from sync_service.services.vector_storage import VectorStorageService
                vector_storage = VectorStorageService()
                qdrant_success = await vector_storage.delete_user_data(user_id)
                results["engines"]["qdrant"] = {
                    "status": "success" if qdrant_success else "error"
                }
            except Exception as e:
                logger.error(f"Erreur lors de la suppression des données Qdrant pour l'utilisateur {user_id}: {e}", exc_info=True)
                results["engines"]["qdrant"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Déterminer le statut global
        all_success = all(engine["status"] == "success" for engine in results["engines"].values())
        any_success = any(engine["status"] == "success" for engine in results["engines"].values())
        
        if all_success:
            results["status"] = "success"
        elif any_success:
            results["status"] = "partial"
        else:
            results["status"] = "error"
        
        logger.info(f"Suppression d'index unifiée terminée pour l'utilisateur {user_id}: {results['status']}")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtient des statistiques sur tous les moteurs.
        
        Returns:
            Dictionnaire contenant des statistiques
        """
        stats = {
            "primary_engine": self.primary_engine_type,
            "usage_stats": self.engine_usage,
            "engines": {
                SearchEngineType.BM25: self.bm25_engine.get_stats(),
                SearchEngineType.WHOOSH: self.whoosh_engine.get_stats()
            },
            "vector_search_available": QDRANT_AVAILABLE
        }
        
        # Ajouter les statistiques Qdrant si disponible
        if QDRANT_AVAILABLE:
            try:
                from sync_service.services.vector_storage import VectorStorageService
                vector_storage = VectorStorageService()
                stats["engines"]["qdrant"] = {"status": "available"}
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des statistiques Qdrant: {e}", exc_info=True)
                stats["engines"]["qdrant"] = {"status": "error", "message": str(e)}
        
        return stats


# Instance globale du moteur unifié
_unified_engine = None

def get_unified_engine() -> UnifiedSearchEngine:
    """
    Obtient l'instance singleton du moteur de recherche unifié.
    
    Returns:
        Instance du moteur unifié
    """
    global _unified_engine
    
    if _unified_engine is None:
        # Déterminer le moteur principal à partir de la configuration
        primary_engine = SearchEngineType(
            os.environ.get("PRIMARY_SEARCH_ENGINE", SearchEngineType.WHOOSH)
        )
        _unified_engine = UnifiedSearchEngine(primary_engine=primary_engine)
    
    return _unified_engine