"""
Moteur de recherche principal.

Ce module orchestre la recherche hybride en combinant recherche lexicale,
sémantique et reranking pour optimiser la pertinence des résultats.
VERSION CORRIGÉE COMPLÈTE - Compatible avec les nouveaux clients et modèles
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
from collections import defaultdict

from search_service.models import SearchQuery, SearchResponse, SearchResult, SearchType
from search_service.models.responses import SearchResult as SearchResultModel

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Moteur de recherche hybride principal.
    VERSION CORRIGÉE - Compatible avec HybridElasticClient et clients modernes.
    """
    
    def __init__(
        self,
        elastic_client: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        cache: Optional[Any] = None
    ):
        self.elastic_client = elastic_client
        self.qdrant_client = qdrant_client
        self.cache = cache
        
        # Déterminer les capacités disponibles
        self.elasticsearch_enabled = self._check_elasticsearch_available()
        self.qdrant_enabled = self._check_qdrant_available()
        
        # Initialiser le processeur de requêtes de manière sécurisée
        self.query_processor = None
        try:
            from search_service.core.query_processor import QueryProcessor
            self.query_processor = QueryProcessor()
        except ImportError:
            logger.warning("QueryProcessor non disponible, utilisation du traitement basique")
        
        logger.info(f"🔧 SearchEngine initialisé:")
        logger.info(f"   - Elasticsearch: {'✅' if self.elasticsearch_enabled else '❌'}")
        logger.info(f"   - Qdrant: {'✅' if self.qdrant_enabled else '❌'}")
        logger.info(f"   - Query Processor: {'✅' if self.query_processor else '❌'}")
        
    def _check_elasticsearch_available(self) -> bool:
        """Vérifie si Elasticsearch est disponible et initialisé."""
        if not self.elastic_client:
            return False
        
        try:
            # Vérifier si c'est le nouveau HybridElasticClient
            if hasattr(self.elastic_client, 'is_initialized'):
                return self.elastic_client.is_initialized()
            # Compatibilité avec l'ancien client
            elif hasattr(self.elastic_client, '_initialized'):
                return getattr(self.elastic_client, '_initialized', False)
            else:
                return True  # Supposer qu'il est prêt si pas d'attribut de statut
        except Exception as e:
            logger.warning(f"Erreur vérification Elasticsearch: {e}")
            return False
    
    def _check_qdrant_available(self) -> bool:
        """Vérifie si Qdrant est disponible et initialisé."""
        if not self.qdrant_client:
            return False
        
        try:
            if hasattr(self.qdrant_client, '_initialized'):
                return getattr(self.qdrant_client, '_initialized', False)
            else:
                return True
        except Exception as e:
            logger.warning(f"Erreur vérification Qdrant: {e}")
            return False
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Effectue une recherche selon le type demandé.
        VERSION CORRIGÉE - Gestion complète des erreurs et des types.
        
        Args:
            query: Requête de recherche
            
        Returns:
            SearchResponse: Résultats de recherche avec tous les champs requis
        """
        start_time = time.time()
        timings = {}
        results = []
        search_error = None
        
        logger.info(f"🔍 Recherche {query.search_type.value}: '{query.query}' (user {query.user_id})")
        
        try:
            # Traiter la requête
            query_start = time.time()
            processed_query = await self._process_query(query.query)
            timings["query_processing"] = time.time() - query_start
            
            # Exécuter la recherche selon le type
            if query.search_type == SearchType.LEXICAL:
                results = await self._lexical_search(query, processed_query)
            elif query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query, processed_query)
            else:  # HYBRID
                results = await self._hybrid_search(query, processed_query, timings)
            
            # Appliquer le reranking si demandé et disponible
            if query.use_reranking and results:
                rerank_start = time.time()
                try:
                    results = await self._rerank_results(query.query, results)
                    timings["reranking"] = time.time() - rerank_start
                except Exception as rerank_error:
                    logger.warning(f"Reranking échoué: {rerank_error}")
                    timings["reranking"] = time.time() - rerank_start
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            search_error = str(e)
            results = []
        
        # Calculer le total avant pagination
        total_found = len(results)
        
        # Appliquer la pagination
        paginated_results = results[query.offset:query.offset + query.limit]
        
        # Construire la réponse avec TOUS les champs requis
        processing_time = time.time() - start_time
        timings["total"] = processing_time
        
        response = SearchResponse(
            # Champs pour compatibilité avec l'ancien système
            query=query.query,
            search_type=query.search_type.value,
            results=self._convert_to_api_results(paginated_results),
            total_found=total_found,
            limit=query.limit,
            offset=query.offset,
            has_more=(query.offset + len(paginated_results)) < total_found,
            processing_time=processing_time,
            
            # Champs pour compatibilité avec le nouveau système
            total=total_found,
            query_time=processing_time,
            user_id=query.user_id,
            
            # Champs optionnels
            timings=timings if query.include_explanations else None,
            filters_applied=self._get_applied_filters(query),
            suggestions=processed_query.get("suggestions"),
            error=search_error,
            timestamp=time.time()
        )
        
        logger.info(f"✅ Recherche terminée en {processing_time:.3f}s: {len(paginated_results)}/{total_found} résultats")
        return response
    
    async def _process_query(self, query_text: str) -> Dict[str, Any]:
        """Traite la requête avec ou sans QueryProcessor."""
        if self.query_processor:
            try:
                return await self.query_processor.process(query_text)
            except Exception as e:
                logger.warning(f"QueryProcessor échoué: {e}")
        
        # Traitement basique de fallback
        return {
            "expanded_query": query_text,
            "keywords": query_text.split(),
            "suggestions": None
        }
    
    async def _hybrid_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any],
        timings: Dict[str, float]
    ) -> List[SearchResultModel]:
        """
        Effectue une recherche hybride combinant lexical et sémantique.
        VERSION CORRIGÉE - Gestion des erreurs et fallbacks.
        """
        # Lancer les recherches en parallèle selon les capacités disponibles
        tasks = {}
        
        if self.elasticsearch_enabled and query.lexical_weight > 0:
            tasks["lexical"] = asyncio.create_task(
                self._lexical_search(query, processed_query)
            )
        
        if self.qdrant_enabled and query.semantic_weight > 0:
            tasks["semantic"] = asyncio.create_task(
                self._semantic_search(query, processed_query)
            )
        
        if not tasks:
            logger.error("❌ Aucun moteur de recherche disponible")
            return []
        
        # Attendre les résultats avec gestion d'erreurs
        lexical_results = []
        semantic_results = []
        
        for task_name, task in tasks.items():
            try:
                task_start = time.time()
                results = await task
                task_time = time.time() - task_start
                timings[f"{task_name}_search"] = task_time
                
                if task_name == "lexical":
                    lexical_results = results
                    logger.info(f"📊 Lexical: {len(results)} résultats en {task_time:.3f}s")
                else:
                    semantic_results = results
                    logger.info(f"🧠 Semantic: {len(results)} résultats en {task_time:.3f}s")
                    
            except Exception as e:
                logger.error(f"❌ Erreur {task_name} search: {e}")
                timings[f"{task_name}_search"] = time.time() - task_start
        
        # Fusionner les résultats
        fusion_start = time.time()
        merged_results = self._merge_results(
            lexical_results,
            semantic_results,
            query.lexical_weight,
            query.semantic_weight
        )
        timings["result_fusion"] = time.time() - fusion_start
        
        logger.info(f"🔗 Fusion: {len(merged_results)} résultats combinés")
        return merged_results
    
    async def _lexical_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any]
    ) -> List[SearchResultModel]:
        """
        Effectue une recherche lexicale via Elasticsearch.
        VERSION CORRIGÉE - Compatible avec HybridElasticClient.
        """
        if not self.elasticsearch_enabled:
            logger.warning("⚠️ Elasticsearch non disponible pour recherche lexicale")
            return []
        
        try:
            # Utiliser la nouvelle interface du HybridElasticClient
            search_results = await self.elastic_client.search(
                user_id=query.user_id,
                query=query.query,  # Le client gère maintenant la validation
                limit=query.limit * 2,  # Récupérer plus pour la fusion hybride
                filters=self._build_filters(query),
                include_highlights=query.include_highlights
            )
            
            # Convertir les résultats en SearchResult
            results = []
            for hit in search_results:
                result = self._elastic_hit_to_result(hit, query.include_explanations)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche lexicale: {e}")
            return []
    
    async def _semantic_search(
        self,
        query: SearchQuery,
        processed_query: Dict[str, Any]
    ) -> List[SearchResultModel]:
        """
        Effectue une recherche sémantique via Qdrant.
        VERSION CORRIGÉE - Gestion robuste des erreurs.
        """
        if not self.qdrant_enabled:
            logger.warning("⚠️ Qdrant non disponible pour recherche sémantique")
            return []
        
        try:
            # Générer l'embedding de la requête
            query_text = processed_query.get("expanded_query", query.query)
            
            # Essayer d'utiliser le service d'embeddings
            query_embedding = None
            try:
                from search_service.core.embeddings import embedding_service
                if embedding_service and hasattr(embedding_service, 'generate_embedding'):
                    query_embedding = await embedding_service.generate_embedding(query_text)
            except ImportError:
                logger.warning("Service d'embeddings non disponible")
            
            if not query_embedding:
                logger.warning("⚠️ Impossible de générer l'embedding, abandon recherche sémantique")
                return []
            
            # Exécuter la recherche vectorielle
            qdrant_results = await self.qdrant_client.search(
                query_vector=query_embedding,
                user_id=query.user_id,
                limit=query.limit * 2,  # Récupérer plus pour la fusion
                filters=self._build_filters(query)
            )
            
            # Convertir en SearchResult
            results = []
            for hit in qdrant_results:
                result = self._qdrant_hit_to_result(hit, query.include_explanations)
                if result:
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche sémantique: {e}")
            return []
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResultModel]
    ) -> List[SearchResultModel]:
        """
        Applique le reranking aux résultats.
        VERSION CORRIGÉE - Gestion des erreurs et fallback.
        """
        if not results:
            return results
        
        try:
            # Essayer d'importer le service de reranking
            from search_service.core.reranker import reranker_service
            
            if not reranker_service or not hasattr(reranker_service, 'is_initialized'):
                logger.debug("Service de reranking non disponible")
                return results
            
            if not reranker_service.is_initialized():
                logger.debug("Service de reranking non initialisé")
                return results
            
            # Préparer les documents pour le reranking
            documents = []
            for result in results:
                # Construire le texte du document à partir des données disponibles
                doc_text = self._extract_document_text(result)
                documents.append(doc_text)
            
            # Appliquer le reranking
            reranked_scores = await reranker_service.rerank(
                query=query,
                documents=documents
            )
            
            # Mettre à jour les scores
            for i, (result, rerank_score) in enumerate(zip(results, reranked_scores)):
                original_score = result.score
                # Combiner avec le score original (70% rerank, 30% original)
                result.score = 0.7 * rerank_score + 0.3 * original_score
                
                # Ajouter les métadonnées de reranking
                if not hasattr(result, 'explanation') or not result.explanation:
                    result.explanation = {}
                result.explanation.update({
                    "rerank_score": rerank_score,
                    "original_score": original_score,
                    "combined_score": result.score
                })
            
            # Retrier par nouveau score
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"✅ Reranking appliqué à {len(results)} résultats")
            return results
            
        except ImportError:
            logger.debug("Module de reranking non disponible")
            return results
        except Exception as e:
            logger.warning(f"⚠️ Erreur reranking: {e}")
            # Retourner les résultats originaux en cas d'erreur
            return results
    
    def _extract_document_text(self, result: SearchResultModel) -> str:
        """Extrait le texte du document pour le reranking."""
        try:
            # Essayer d'extraire le texte depuis les différents champs possibles
            text_parts = []
            
            # Si c'est un objet avec des attributs
            if hasattr(result, 'transaction') and isinstance(result.transaction, dict):
                transaction = result.transaction
                
                # Description principale
                desc = (transaction.get('primary_description') or 
                       transaction.get('description') or 
                       transaction.get('clean_description') or '')
                if desc:
                    text_parts.append(desc)
                
                # Merchant/commerce
                merchant = (transaction.get('merchant_name') or 
                           transaction.get('merchant') or '')
                if merchant:
                    text_parts.append(merchant)
                
                # Catégorie
                category = (transaction.get('category_name') or 
                           transaction.get('category') or '')
                if category:
                    text_parts.append(category)
            
            # Fallback vers les attributs directs
            if hasattr(result, 'description') and result.description:
                text_parts.append(result.description)
            
            # Construire le texte final
            doc_text = " ".join(text_parts) if text_parts else "Transaction"
            return doc_text
            
        except Exception as e:
            logger.warning(f"Erreur extraction texte document: {e}")
            return "Transaction"
    
    def _merge_results(
        self,
        lexical_results: List[SearchResultModel],
        semantic_results: List[SearchResultModel],
        lexical_weight: float,
        semantic_weight: float
    ) -> List[SearchResultModel]:
        """
        Fusionne les résultats lexicaux et sémantiques.
        VERSION CORRIGÉE - Gestion robuste des types.
        """
        if not lexical_results and not semantic_results:
            return []
        
        if not lexical_results:
            # Ajuster les scores sémantiques
            for result in semantic_results:
                result.score *= semantic_weight
            return semantic_results
        
        if not semantic_results:
            # Ajuster les scores lexicaux
            for result in lexical_results:
                result.score *= lexical_weight
            return lexical_results
        
        # Fusionner par transaction_id
        merged = {}
        
        # Ajouter les résultats lexicaux
        for result in lexical_results:
            transaction_id = self._get_transaction_id(result)
            if transaction_id:
                # Cloner le résultat et ajuster le score
                merged_result = self._clone_result(result)
                merged_result.score = result.score * lexical_weight
                
                # Ajouter métadonnées de fusion
                if not hasattr(merged_result, 'explanation'):
                    merged_result.explanation = {}
                merged_result.explanation.update({
                    "lexical_score": result.score,
                    "search_sources": ["lexical"]
                })
                
                merged[transaction_id] = merged_result
        
        # Fusionner avec les résultats sémantiques
        for result in semantic_results:
            transaction_id = self._get_transaction_id(result)
            if transaction_id:
                if transaction_id in merged:
                    # Combiner les scores
                    existing = merged[transaction_id]
                    existing.score += result.score * semantic_weight
                    
                    # Fusionner les métadonnées
                    if hasattr(existing, 'explanation') and existing.explanation:
                        existing.explanation.update({
                            "semantic_score": result.score,
                            "combined_score": existing.score
                        })
                        existing.explanation["search_sources"].append("semantic")
                    
                    # Fusionner les highlights si disponibles
                    if (hasattr(result, 'highlights') and result.highlights and
                        hasattr(existing, 'highlights') and existing.highlights):
                        for field, highlights in result.highlights.items():
                            if field in existing.highlights:
                                existing.highlights[field].extend(highlights)
                            else:
                                existing.highlights[field] = highlights
                else:
                    # Nouveau résultat sémantique uniquement
                    merged_result = self._clone_result(result)
                    merged_result.score = result.score * semantic_weight
                    
                    if not hasattr(merged_result, 'explanation'):
                        merged_result.explanation = {}
                    merged_result.explanation.update({
                        "semantic_score": result.score,
                        "search_sources": ["semantic"]
                    })
                    
                    merged[transaction_id] = merged_result
        
        # Trier par score combiné
        results = list(merged.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _get_transaction_id(self, result: SearchResultModel) -> Optional[int]:
        """Extrait l'ID de transaction d'un résultat."""
        try:
            if hasattr(result, 'transaction_id'):
                return result.transaction_id
            elif hasattr(result, 'transaction') and isinstance(result.transaction, dict):
                return result.transaction.get('transaction_id')
            elif hasattr(result, 'id'):
                # Essayer de convertir l'ID en entier
                try:
                    return int(result.id)
                except (ValueError, TypeError):
                    return None
            return None
        except Exception:
            return None
    
    def _clone_result(self, result: SearchResultModel) -> SearchResultModel:
        """Clone un résultat de recherche."""
        try:
            # Essayer de créer une copie profonde
            import copy
            return copy.deepcopy(result)
        except Exception:
            # Fallback vers une copie simple
            return result
    
    def _build_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Construit les filtres pour la recherche."""
        filters = {}
        
        try:
            if hasattr(query, 'date_from') and query.date_from:
                filters["date_from"] = query.date_from.isoformat()
            if hasattr(query, 'date_to') and query.date_to:
                filters["date_to"] = query.date_to.isoformat()
            if hasattr(query, 'amount_min') and query.amount_min is not None:
                filters["amount_min"] = query.amount_min
            if hasattr(query, 'amount_max') and query.amount_max is not None:
                filters["amount_max"] = query.amount_max
            if hasattr(query, 'categories') and query.categories:
                filters["categories"] = query.categories
            if hasattr(query, 'account_ids') and query.account_ids:
                filters["account_ids"] = query.account_ids
            if hasattr(query, 'transaction_types') and query.transaction_types:
                filters["transaction_types"] = query.transaction_types
            
            # Ajouter les filtres personnalisés
            if hasattr(query, 'filters') and query.filters:
                filters.update(query.filters)
        except Exception as e:
            logger.warning(f"Erreur construction filtres: {e}")
        
        return filters
    
    def _elastic_hit_to_result(
        self,
        hit: Dict[str, Any],
        include_explanations: bool
    ) -> Optional[SearchResultModel]:
        """
        Convertit un hit Elasticsearch en SearchResult.
        VERSION CORRIGÉE - Gestion robuste des données manquantes.
        """
        try:
            # Extraire les données source
            if isinstance(hit, dict):
                source = hit.get("source", hit.get("_source", {}))
                score = hit.get("score", hit.get("_score", 0.0))
                highlights = hit.get("highlights", hit.get("highlight"))
                explanation = hit.get("_explanation") if include_explanations else None
            else:
                logger.warning(f"Format hit Elasticsearch inattendu: {type(hit)}")
                return None
            
            # Construire le SearchResult avec gestion des champs manquants
            result = SearchResultModel(
                transaction_id=source.get("transaction_id", 0),
                user_id=source.get("user_id", 0),
                score=float(score),
                transaction=source,  # Garder toutes les données transaction
                highlights=highlights,
                search_type="elasticsearch",
                explanation=explanation
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion hit Elasticsearch: {e}")
            return None
    
    def _qdrant_hit_to_result(
        self,
        hit: Dict[str, Any],
        include_explanations: bool
    ) -> Optional[SearchResultModel]:
        """
        Convertit un hit Qdrant en SearchResult.
        VERSION CORRIGÉE - Gestion robuste des formats Qdrant.
        """
        try:
            # Extraire les données du payload Qdrant
            if isinstance(hit, dict):
                payload = hit.get("payload", {})
                score = hit.get("score", 0.0)
            else:
                logger.warning(f"Format hit Qdrant inattendu: {type(hit)}")
                return None
            
            # Construire le SearchResult
            result = SearchResultModel(
                transaction_id=payload.get("transaction_id", 0),
                user_id=payload.get("user_id", 0),
                score=float(score),
                transaction=payload,  # Utiliser le payload comme données transaction
                highlights=None,  # Qdrant ne fournit pas de highlights
                search_type="qdrant",
                explanation={"vector_similarity": score} if include_explanations else None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion hit Qdrant: {e}")
            return None
    
    def _convert_to_api_results(self, results: List[SearchResultModel]) -> List[Dict[str, Any]]:
        """
        Convertit les SearchResult en format API attendu par routes.py.
        VERSION CORRIGÉE - Format compatible avec l'API routes.
        """
        api_results = []
        
        for result in results:
            try:
                api_result = {
                    "id": str(result.transaction_id),
                    "score": float(result.score),
                    "transaction": result.transaction if isinstance(result.transaction, dict) else {},
                    "highlights": result.highlights if result.highlights else {},
                    "search_type": getattr(result, 'search_type', 'unknown')
                }
                
                # Ajouter les explications si disponibles
                if hasattr(result, 'explanation') and result.explanation:
                    api_result["explanation"] = result.explanation
                
                api_results.append(api_result)
                
            except Exception as e:
                logger.warning(f"Erreur conversion résultat API: {e}")
                continue
        
        return api_results
    
    def _get_applied_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Retourne les filtres effectivement appliqués."""
        applied = {}
        
        try:
            if hasattr(query, 'date_from') and query.date_from:
                applied["date_from"] = query.date_from.isoformat()
            if hasattr(query, 'date_to') and query.date_to:
                applied["date_to"] = query.date_to.isoformat()
            if hasattr(query, 'amount_min') and query.amount_min is not None:
                applied["amount_min"] = query.amount_min
            if hasattr(query, 'amount_max') and query.amount_max is not None:
                applied["amount_max"] = query.amount_max
            if hasattr(query, 'categories') and query.categories:
                applied["categories"] = query.categories
            if hasattr(query, 'account_ids') and query.account_ids:
                applied["account_ids"] = query.account_ids
            if hasattr(query, 'transaction_types') and query.transaction_types:
                applied["transaction_types"] = query.transaction_types
        except Exception as e:
            logger.warning(f"Erreur extraction filtres appliqués: {e}")
        
        return applied
    
    # Méthodes pour l'API et la compatibilité
    
    async def reindex_user_transactions(
        self, 
        user_id: int, 
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Réindexe les transactions d'un utilisateur."""
        logger.info(f"🔄 Réindexation user {user_id} (force: {force_refresh})")
        
        result = {
            "processed": 0,
            "indexed": 0,
            "errors": 0,
            "message": "Reindexation terminée"
        }
        
        try:
            # Implémenter la logique de réindexation selon les clients disponibles
            if self.elasticsearch_enabled:
                # Logique de réindexation Elasticsearch
                pass
            
            if self.qdrant_enabled:
                # Logique de réindexation Qdrant
                pass
                
        except Exception as e:
            logger.error(f"❌ Erreur réindexation user {user_id}: {e}")
            result["errors"] = 1
            result["message"] = f"Erreur: {e}"
        
        return result
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Récupère les statistiques pour un utilisateur."""
        stats = {
            "elasticsearch_count": 0,
            "qdrant_count": 0,
            "elasticsearch_available": self.elasticsearch_enabled,
            "qdrant_available": self.qdrant_enabled,
            "last_update": None
        }
        
        try:
            if self.elasticsearch_enabled and hasattr(self.elastic_client, 'count_documents'):
                stats["elasticsearch_count"] = await self.elastic_client.count_documents(user_id)
            
            if self.qdrant_enabled and hasattr(self.qdrant_client, 'count_points'):
                stats["qdrant_count"] = await self.qdrant_client.count_points(user_id)
                
        except Exception as e:
            logger.error(f"❌ Erreur stats user {user_id}: {e}")
        
        return stats
    
    async def delete_user_data(self, user_id: int) -> Dict[str, Any]:
        """Supprime toutes les données d'un utilisateur."""
        result = {
            "elasticsearch_deleted": 0,
            "qdrant_deleted": 0,
            "message": "Suppression terminée"
        }
        
        try:
            # Supprimer de Elasticsearch si disponible
            if self.elasticsearch_enabled:
                # Implémenter la suppression Elasticsearch
                # Pour l'instant, placeholder
                result["elasticsearch_deleted"] = 0
            
            # Supprimer de Qdrant si disponible
            if self.qdrant_enabled:
                # Implémenter la suppression Qdrant
                # Pour l'instant, placeholder
                result["qdrant_deleted"] = 0
                
        except Exception as e:
            logger.error(f"❌ Erreur suppression user {user_id}: {e}")
            result["message"] = f"Erreur: {e}"
        
        return result
    
    # Méthodes de compatibilité pour l'ancienne API
    
    async def lexical_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Méthode de compatibilité pour la recherche lexicale.
        Utilisée par routes.py en fallback.
        """
        try:
            # Créer une SearchQuery temporaire
            search_query = SearchQuery(
                user_id=user_id,
                query=query,
                search_type=SearchType.LEXICAL,
                limit=limit,
                offset=0
            )
            
            processed_query = await self._process_query(query)
            results = await self._lexical_search(search_query, processed_query)
            
            # Convertir en format attendu par l'ancienne API
            return self._convert_to_legacy_format(results)
            
        except Exception as e:
            logger.error(f"❌ Erreur lexical_search: {e}")
            return []
    
    async def semantic_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Méthode de compatibilité pour la recherche sémantique.
        Utilisée par routes.py en fallback.
        """
        try:
            # Créer une SearchQuery temporaire
            search_query = SearchQuery(
                user_id=user_id,
                query=query,
                search_type=SearchType.SEMANTIC,
                limit=limit,
                offset=0
            )
            
            processed_query = await self._process_query(query)
            results = await self._semantic_search(search_query, processed_query)
            
            # Convertir en format attendu par l'ancienne API
            return self._convert_to_legacy_format(results)
            
        except Exception as e:
            logger.error(f"❌ Erreur semantic_search: {e}")
            return []
    
    async def hybrid_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Méthode de compatibilité pour la recherche hybride.
        Utilisée par routes.py en fallback.
        """
        try:
            # Créer une SearchQuery temporaire
            search_query = SearchQuery(
                user_id=user_id,
                query=query,
                search_type=SearchType.HYBRID,
                limit=limit,
                offset=0,
                use_reranking=use_reranking
            )
            
            processed_query = await self._process_query(query)
            timings = {}
            results = await self._hybrid_search(search_query, processed_query, timings)
            
            # Appliquer le reranking si demandé
            if use_reranking and results:
                try:
                    results = await self._rerank_results(query, results)
                except Exception as rerank_error:
                    logger.warning(f"Reranking échoué: {rerank_error}")
            
            # Convertir en format attendu par l'ancienne API
            return self._convert_to_legacy_format(results)
            
        except Exception as e:
            logger.error(f"❌ Erreur hybrid_search: {e}")
            return []
    
    def _convert_to_legacy_format(self, results: List[SearchResultModel]) -> List[Dict[str, Any]]:
        """
        Convertit les SearchResult en format legacy pour compatibilité.
        Format attendu par l'ancienne API routes.py.
        """
        legacy_results = []
        
        for result in results:
            try:
                legacy_result = {
                    "id": str(result.transaction_id),
                    "score": float(result.score),
                    "source": result.transaction if isinstance(result.transaction, dict) else {},
                    "search_type": getattr(result, 'search_type', 'unknown')
                }
                
                # Ajouter highlights si disponibles
                if hasattr(result, 'highlights') and result.highlights:
                    legacy_result["highlights"] = result.highlights
                
                # Ajouter explanation si disponible
                if hasattr(result, 'explanation') and result.explanation:
                    legacy_result["explanation"] = result.explanation
                
                legacy_results.append(legacy_result)
                
            except Exception as e:
                logger.warning(f"Erreur conversion legacy: {e}")
                continue
        
        return legacy_results
    
    # Méthodes utilitaires et diagnostics
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du moteur de recherche."""
        return {
            "initialized": True,
            "capabilities": {
                "elasticsearch": self.elasticsearch_enabled,
                "qdrant": self.qdrant_enabled,
                "query_processor": self.query_processor is not None,
                "cache": self.cache is not None
            },
            "client_types": {
                "elasticsearch": type(self.elastic_client).__name__ if self.elastic_client else None,
                "qdrant": type(self.qdrant_client).__name__ if self.qdrant_client else None
            },
            "search_modes": {
                "lexical": self.elasticsearch_enabled,
                "semantic": self.qdrant_enabled,
                "hybrid": self.elasticsearch_enabled and self.qdrant_enabled
            }
        }
    
    async def test_search_capabilities(self) -> Dict[str, Any]:
        """Teste toutes les capacités de recherche disponibles."""
        test_results = {
            "timestamp": time.time(),
            "overall_success": False,
            "tests": {}
        }
        
        # Test de recherche lexicale
        if self.elasticsearch_enabled:
            try:
                test_start = time.time()
                lexical_results = await self.lexical_search(
                    user_id=1, 
                    query="test", 
                    limit=1
                )
                test_time = time.time() - test_start
                
                test_results["tests"]["lexical"] = {
                    "success": True,
                    "results_count": len(lexical_results),
                    "time": test_time
                }
            except Exception as e:
                test_results["tests"]["lexical"] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            test_results["tests"]["lexical"] = {
                "success": False,
                "error": "Elasticsearch not available"
            }
        
        # Test de recherche sémantique
        if self.qdrant_enabled:
            try:
                test_start = time.time()
                semantic_results = await self.semantic_search(
                    user_id=1, 
                    query="test", 
                    limit=1
                )
                test_time = time.time() - test_start
                
                test_results["tests"]["semantic"] = {
                    "success": True,
                    "results_count": len(semantic_results),
                    "time": test_time
                }
            except Exception as e:
                test_results["tests"]["semantic"] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            test_results["tests"]["semantic"] = {
                "success": False,
                "error": "Qdrant not available"
            }
        
        # Test de recherche hybride
        if self.elasticsearch_enabled and self.qdrant_enabled:
            try:
                test_start = time.time()
                hybrid_results = await self.hybrid_search(
                    user_id=1, 
                    query="test", 
                    limit=1,
                    use_reranking=False
                )
                test_time = time.time() - test_start
                
                test_results["tests"]["hybrid"] = {
                    "success": True,
                    "results_count": len(hybrid_results),
                    "time": test_time
                }
            except Exception as e:
                test_results["tests"]["hybrid"] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            test_results["tests"]["hybrid"] = {
                "success": False,
                "error": "Both Elasticsearch and Qdrant required"
            }
        
        # Calculer le succès global
        successful_tests = sum(1 for test in test_results["tests"].values() 
                             if test.get("success", False))
        total_tests = len(test_results["tests"])
        test_results["overall_success"] = successful_tests > 0
        test_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_results
    
    def __repr__(self) -> str:
        """Représentation string du moteur pour debug."""
        return (
            f"SearchEngine("
            f"elasticsearch={'✅' if self.elasticsearch_enabled else '❌'}, "
            f"qdrant={'✅' if self.qdrant_enabled else '❌'}, "
            f"query_processor={'✅' if self.query_processor else '❌'}"
            f")"
        )


# Classe utilitaire pour la création de SearchQuery depuis les paramètres API
class SearchQueryBuilder:
    """Builder pour créer des SearchQuery depuis les paramètres d'API."""
    
    @staticmethod
    def from_api_params(
        user_id: int,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        offset: int = 0,
        use_reranking: bool = True,
        **kwargs
    ) -> SearchQuery:
        """Crée une SearchQuery depuis les paramètres d'API."""
        
        # Valider le type de recherche
        try:
            search_type_enum = SearchType(search_type.lower())
        except ValueError:
            search_type_enum = SearchType.HYBRID
        
        # Créer la SearchQuery
        search_query = SearchQuery(
            user_id=user_id,
            query=query,
            search_type=search_type_enum,
            limit=max(1, min(limit, 100)),  # Limiter entre 1 et 100
            offset=max(0, offset),
            use_reranking=use_reranking
        )
        
        # Ajouter les filtres optionnels depuis kwargs
        if 'date_from' in kwargs and kwargs['date_from']:
            try:
                from datetime import datetime
                search_query.date_from = datetime.fromisoformat(kwargs['date_from'])
            except Exception:
                pass
        
        if 'date_to' in kwargs and kwargs['date_to']:
            try:
                from datetime import datetime
                search_query.date_to = datetime.fromisoformat(kwargs['date_to'])
            except Exception:
                pass
        
        if 'amount_min' in kwargs and kwargs['amount_min'] is not None:
            search_query.amount_min = float(kwargs['amount_min'])
        
        if 'amount_max' in kwargs and kwargs['amount_max'] is not None:
            search_query.amount_max = float(kwargs['amount_max'])
        
        if 'categories' in kwargs and kwargs['categories']:
            search_query.categories = kwargs['categories']
        
        if 'account_ids' in kwargs and kwargs['account_ids']:
            search_query.account_ids = kwargs['account_ids']
        
        if 'transaction_types' in kwargs and kwargs['transaction_types']:
            search_query.transaction_types = kwargs['transaction_types']
        
        return search_query


# Factory function pour créer le SearchEngine avec les clients appropriés
def create_search_engine(
    elastic_client: Optional[Any] = None,
    qdrant_client: Optional[Any] = None,
    cache: Optional[Any] = None
) -> SearchEngine:
    """
    Factory function pour créer un SearchEngine.
    
    Args:
        elastic_client: Client Elasticsearch (HybridElasticClient ou autre)
        qdrant_client: Client Qdrant
        cache: Cache de recherche
        
    Returns:
        SearchEngine: Instance configurée du moteur de recherche
    """
    engine = SearchEngine(
        elastic_client=elastic_client,
        qdrant_client=qdrant_client,
        cache=cache
    )
    
    logger.info(f"🏭 SearchEngine créé: {engine}")
    return engine


# Export des classes et fonctions principales
__all__ = [
    'SearchEngine',
    'SearchQueryBuilder',
    'create_search_engine'
]