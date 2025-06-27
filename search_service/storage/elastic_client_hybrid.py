"""
Client Elasticsearch hybride qui utilise le client officiel ou Bonsai HTTP selon la compatibilité.
VERSION CORRIGÉE COMPLÈTE - Résout tous les bugs de validation et de type
"""
import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union

from config_service.config import settings

logger = logging.getLogger("search_service.elasticsearch")
metrics_logger = logging.getLogger("search_service.metrics.elasticsearch")


class HybridElasticClient:
    """
    Client hybride qui choisit automatiquement entre Elasticsearch officiel et Bonsai HTTP.
    Résout les problèmes de compatibilité Bonsai et de validation des types.
    """
    
    def __init__(self):
        self.client = None
        self.bonsai_client = None
        self.client_type = None  # 'elasticsearch' ou 'bonsai'
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        self._closed = False
        self._initialization_error = None
        
    async def initialize(self) -> bool:
        """Initialise la connexion en essayant d'abord Elasticsearch, puis Bonsai."""
        logger.info("🔄 Initialisation du client Elasticsearch hybride...")
        start_time = time.time()
        
        # Reset des états précédents
        self._initialization_error = None
        self._connection_attempts = 0
        
        if not settings.BONSAI_URL:
            error_msg = "BONSAI_URL non configurée"
            logger.error(f"❌ {error_msg}")
            self._initialization_error = error_msg
            return False
        
        # Fermer les clients existants
        await self._cleanup_existing_clients()
        
        # Masquer les credentials pour l'affichage
        safe_url = self._mask_credentials(settings.BONSAI_URL)
        logger.info(f"🔗 Connexion à: {safe_url}")
        
        # Essayer d'abord le client Elasticsearch officiel
        elasticsearch_success = await self._try_elasticsearch_client()
        
        if elasticsearch_success:
            self.client_type = 'elasticsearch'
            self._initialized = True
            self._closed = False
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Elasticsearch officiel initialisé en {total_time:.2f}s")
            return True
        
        # Si échec, essayer le client Bonsai HTTP
        logger.info("🔄 Tentative avec client Bonsai HTTP...")
        bonsai_success = await self._try_bonsai_client()
        
        if bonsai_success:
            self.client_type = 'bonsai'
            self._initialized = True
            self._closed = False
            total_time = time.time() - start_time
            logger.info(f"🎉 Client Bonsai HTTP initialisé en {total_time:.2f}s")
            return True
        
        # Les deux ont échoué
        error_msg = "Impossible d'initialiser un client de recherche"
        logger.error(f"❌ {error_msg}")
        self._initialization_error = error_msg
        return False
    
    async def _cleanup_existing_clients(self):
        """Nettoie les clients existants proprement."""
        cleanup_tasks = []
        
        if self.client:
            logger.debug("🧹 Nettoyage client Elasticsearch existant...")
            try:
                # Utiliser _safe_close_elasticsearch pour éviter les duplications
                cleanup_tasks.append(self._safe_close_elasticsearch())
            except Exception as e:
                logger.warning(f"⚠️ Erreur préparation nettoyage Elasticsearch: {e}")
                self.client = None
        
        if self.bonsai_client:
            logger.debug("🧹 Nettoyage client Bonsai existant...")
            try:
                if hasattr(self.bonsai_client, 'close'):
                    cleanup_tasks.append(self.bonsai_client.close())
                else:
                    self.bonsai_client = None
            except Exception as e:
                logger.warning(f"⚠️ Erreur préparation nettoyage Bonsai: {e}")
                self.bonsai_client = None
        
        # Attendre tous les nettoyages avec timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout nettoyage clients")
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage groupé: {e}")
    
    async def _try_elasticsearch_client(self) -> bool:
        """Essaie d'initialiser le client Elasticsearch officiel."""
        try:
            from elasticsearch import AsyncElasticsearch
            
            logger.info("🔍 Tentative avec client Elasticsearch officiel...")
            self._connection_attempts += 1
            
            self.client = AsyncElasticsearch(
                [settings.BONSAI_URL],
                verify_certs=True,
                ssl_show_warn=False,
                max_retries=2,  # Réduire pour éviter les timeouts longs
                retry_on_timeout=True,
                request_timeout=15.0,  # Timeout plus court
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Test de connexion avec timeout strict
            try:
                info = await asyncio.wait_for(self.client.info(), timeout=10.0)
                cluster_name = info.get('cluster_name', 'unknown')
                version = info.get('version', {}).get('number', 'unknown')
                logger.info(f"✅ Client Elasticsearch: {cluster_name} v{version}")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout connexion Elasticsearch (10s)")
                await self._safe_close_elasticsearch()
                return False
            except Exception as conn_error:
                logger.warning(f"⚠️ Erreur connexion Elasticsearch: {conn_error}")
                await self._safe_close_elasticsearch()
                return False
            
            # Test de santé rapide
            try:
                health = await asyncio.wait_for(self.client.cluster.health(), timeout=5.0)
                status = health.get('status', 'unknown')
                logger.info(f"💚 Santé cluster: {status}")
                
                # Tester l'index si possible
                try:
                    index_exists = await asyncio.wait_for(
                        self.client.indices.exists(index=self.index_name),
                        timeout=3.0
                    )
                    logger.info(f"📁 Index {self.index_name}: {'✅ existe' if index_exists else '❌ manquant'}")
                except Exception:
                    logger.debug("📁 Impossible de vérifier l'existence de l'index")
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout health check Elasticsearch")
            except Exception as health_error:
                logger.warning(f"⚠️ Erreur health check: {health_error}")
            
            return True
            
        except ImportError:
            logger.error("❌ Module elasticsearch non disponible")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Client Elasticsearch échoué: {e}")
            
            # Détection spécifique de l'incompatibilité Bonsai
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in [
                "unsupportedproducterror", 
                "not elasticsearch", 
                "product check", 
                "opensearch"
            ]):
                logger.info("💡 Bonsai détecté comme incompatible avec le client standard")
            
            await self._safe_close_elasticsearch()
            return False
    
    async def _safe_close_elasticsearch(self):
        """Ferme le client Elasticsearch de manière sécurisée."""
        if self.client:
            try:
                await asyncio.wait_for(self.client.close(), timeout=5.0)
                logger.debug("🔒 Client Elasticsearch fermé proprement")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout fermeture Elasticsearch")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture Elasticsearch: {e}")
            finally:
                self.client = None
    
    async def _try_bonsai_client(self) -> bool:
        """Essaie d'initialiser le client Bonsai HTTP."""
        try:
            # Import dynamique pour éviter les erreurs si le module n'existe pas
            try:
                from .bonsai_client import BonsaiClient
            except ImportError:
                logger.error("❌ Module bonsai_client non disponible")
                return False
            
            logger.info("🌐 Tentative avec client Bonsai HTTP...")
            self._connection_attempts += 1
            
            self.bonsai_client = BonsaiClient()
            success = await self.bonsai_client.initialize()
            
            if success:
                logger.info("✅ Client Bonsai HTTP opérationnel")
                return True
            else:
                logger.error("❌ Client Bonsai HTTP échoué à l'initialisation")
                self.bonsai_client = None
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur client Bonsai: {e}")
            self.bonsai_client = None
            return False
    
    def _mask_credentials(self, url: str) -> str:
        """Masque les credentials dans l'URL pour les logs."""
        if not url or not isinstance(url, str):
            return "URL invalide"
            
        try:
            if "@" in url:
                parts = url.split("@")
                if len(parts) == 2:
                    protocol_and_creds = parts[0]
                    host_and_path = parts[1]
                    
                    if "://" in protocol_and_creds:
                        protocol = protocol_and_creds.split("://")[0]
                        return f"{protocol}://***:***@{host_and_path}"
            
            return url
        except Exception:
            return "URL non analysable"
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client est sain et fonctionnel."""
        if not self._initialized or self._closed:
            return False
        
        try:
            current_time = time.time()
            
            if self.client_type == 'elasticsearch' and self.client:
                health = await asyncio.wait_for(self.client.cluster.health(), timeout=5.0)
                status = health.get("status", "red")
                is_healthy = status in ["green", "yellow"]
                
                self._last_health_check = {
                    "timestamp": current_time,
                    "healthy": is_healthy,
                    "status": status,
                    "client_type": "elasticsearch"
                }
                
                return is_healthy
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                is_healthy = await self.bonsai_client.is_healthy()
                
                self._last_health_check = {
                    "timestamp": current_time,
                    "healthy": is_healthy,
                    "status": "ok" if is_healthy else "error",
                    "client_type": "bonsai"
                }
                
                return is_healthy
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            self._last_health_check = {
                "timestamp": time.time(),
                "healthy": False,
                "status": "error",
                "client_type": self.client_type,
                "error": str(e)
            }
            return False
    
    def _validate_query_input(self, query: Any) -> str:
        """
        Valide et nettoie l'input de requête.
        RÉSOUT LE BUG: 'dict' object has no attribute 'lower'
        """
        if query is None:
            return ""
        
        # Si c'est déjà une string, la nettoyer
        if isinstance(query, str):
            return query.strip()
        
        # Si c'est un dictionnaire, extraire le contenu pertinent
        if isinstance(query, dict):
            logger.warning(f"⚠️ Query reçue comme dict: {query}")
            
            # Essayer d'extraire une string du dictionnaire
            possible_keys = ['query', 'q', 'search', 'text', 'term']
            for key in possible_keys:
                if key in query and isinstance(query[key], str):
                    extracted = query[key].strip()
                    logger.info(f"📤 Query extraite du dict['{key}']: '{extracted}'")
                    return extracted
            
            # Si pas de clé pertinente, convertir en JSON
            try:
                json_str = json.dumps(query)
                logger.warning(f"📦 Dict converti en JSON: {json_str}")
                return json_str
            except Exception as e:
                logger.error(f"❌ Impossible de traiter le dict: {e}")
                return str(query)
        
        # Pour tout autre type, convertir en string
        try:
            converted = str(query).strip()
            logger.info(f"🔄 Query convertie de {type(query).__name__} vers string: '{converted}'")
            return converted
        except Exception as e:
            logger.error(f"❌ Impossible de convertir query {type(query)}: {e}")
            return ""
    
    async def search(
        self,
        user_id: int,
        query: Union[str, dict, Any],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recherche des transactions via le client approprié.
        VERSION CORRIGÉE - Gestion robuste des types de query.
        """
        if not self._initialized or self._closed:
            logger.error("❌ Client non initialisé ou fermé")
            return []
        
        # VALIDATION CRITIQUE: Nettoyer et valider la query
        validated_query = self._validate_query_input(query)
        
        if not validated_query:
            logger.warning("⚠️ Query vide après validation")
            return []
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche via {self.client_type}")
        logger.info(f"    Query: '{validated_query}' (validée depuis {type(query).__name__})")
        logger.info(f"    User: {user_id}, Limit: {limit}")
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                return await self._search_elasticsearch(
                    search_id, user_id, validated_query, limit, filters, include_highlights
                )
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.search(
                    user_id, validated_query, limit, filters, include_highlights
                )
            else:
                logger.error("❌ Aucun client disponible pour la recherche")
                return []
                
        except Exception as e:
            logger.error(f"❌ [{search_id}] Erreur de recherche: {e}")
            logger.error(f"    Query originale: {type(query).__name__} = {query}")
            logger.error(f"    Query validée: '{validated_query}'")
            return []
    
    async def _search_elasticsearch(
        self,
        search_id: str,
        user_id: int,
        query: str,  # Maintenant garantie d'être une string
        limit: int,
        filters: Optional[Dict[str, Any]],
        include_highlights: bool
    ) -> List[Dict[str, Any]]:
        """
        Recherche avec le client Elasticsearch officiel.
        VERSION CORRIGÉE - Query garantie string.
        """
        start_time = time.time()
        
        try:
            # Expansion des termes de recherche sécurisée
            expanded_terms = []
            search_string = query  # Fallback par défaut
            
            try:
                from search_service.utils.query_expansion import expand_query_terms
                expanded_terms = expand_query_terms(query)
                
                # Validation des termes expandus
                validated_terms = []
                for term in expanded_terms:
                    if isinstance(term, str) and term.strip():
                        validated_terms.append(term.strip())
                    elif term is not None:
                        term_str = str(term).strip()
                        if term_str:
                            validated_terms.append(term_str)
                
                expanded_terms = validated_terms
                search_string = " ".join(expanded_terms) if expanded_terms else query
                
            except ImportError:
                logger.debug("Module query_expansion non disponible, utilisation query directe")
                expanded_terms = [query]
                search_string = query
            except Exception as exp_error:
                logger.warning(f"⚠️ Erreur expansion query: {exp_error}")
                expanded_terms = [query]
                search_string = query
            
            # Construction de la requête Elasticsearch sécurisée
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}}
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": search_string,  # String garantie
                                    "fields": [
                                        "searchable_text^3", 
                                        "primary_description^2", 
                                        "merchant_name^2", 
                                        "category_name"
                                    ],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": min(max(limit, 1), 100),  # Limiter entre 1 et 100
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"transaction_date": {"order": "desc", "unmapped_type": "date"}}
                ]
            }
            
            # Ajouter terms query seulement si on a des termes valides
            if expanded_terms and len(expanded_terms) > 0:
                search_body["query"]["bool"]["should"].append({
                    "terms": {
                        "primary_description.keyword": expanded_terms,
                        "boost": 2.0
                    }
                })
            
            # Ajouter les filtres
            if filters and isinstance(filters, dict):
                if "filter" not in search_body["query"]["bool"]:
                    search_body["query"]["bool"]["filter"] = []
                
                for field, value in filters.items():
                    if value is not None and field:
                        search_body["query"]["bool"]["filter"].append({
                            "term": {field: value}
                        })
            
            # Ajouter highlighting
            if include_highlights:
                search_body["highlight"] = {
                    "fields": {
                        "searchable_text": {"fragment_size": 150},
                        "primary_description": {"fragment_size": 100},
                        "merchant_name": {"fragment_size": 50}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "max_analyzed_offset": 60000
                }
            
            logger.debug(f"🎯 [{search_id}] Query Elasticsearch construite")
            
            # Exécuter la recherche avec timeout
            response = await asyncio.wait_for(
                self.client.search(index=self.index_name, body=search_body),
                timeout=20.0
            )
            
            query_time = time.time() - start_time
            
            # Traiter les résultats
            hits = response.get("hits", {}).get("hits", [])
            total_hits = response.get("hits", {}).get("total", {})
            
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
            else:
                total_count = int(total_hits) if total_hits else 0
            
            results = []
            scores = []
            
            for hit in hits:
                result_item = {
                    "id": hit["_id"],
                    "score": float(hit["_score"]),
                    "source": hit["_source"],
                    "search_type": "elasticsearch"
                }
                
                if "highlight" in hit and include_highlights:
                    result_item["highlights"] = hit["highlight"]
                
                results.append(result_item)
                scores.append(hit["_score"])
            
            # Logs de résultats
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
            else:
                max_score = min_score = avg_score = 0.0
            
            logger.info(f"✅ [{search_id}] Recherche terminée en {query_time:.3f}s")
            logger.info(f"📊 [{search_id}] Résultats: {len(results)}/{total_count}")
            logger.info(f"🎯 [{search_id}] Scores: max={max_score:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
            
            # Métriques pour monitoring
            metrics_logger.info(
                f"elasticsearch.search.success,"
                f"user_id={user_id},"
                f"query_time={query_time:.3f},"
                f"results={len(results)},"
                f"total={total_count},"
                f"max_score={max_score:.3f}"
            )
            
            return results
            
        except asyncio.TimeoutError:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Timeout Elasticsearch après {query_time:.3f}s")
            metrics_logger.error(f"elasticsearch.search.timeout,user_id={user_id},time={query_time:.3f}")
            return []
            
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur Elasticsearch après {query_time:.3f}s: {e}")
            metrics_logger.error(f"elasticsearch.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document via le client approprié."""
        if not self._initialized or self._closed:
            logger.error("❌ Client non initialisé pour indexation")
            return False
        
        if not doc_id or not isinstance(document, dict):
            logger.error("❌ Paramètres invalides pour indexation")
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                response = await asyncio.wait_for(
                    self.client.index(index=self.index_name, id=doc_id, body=document),
                    timeout=15.0
                )
                result = response.get("result", "").lower()
                success = result in ["created", "updated"]
                
                if success:
                    logger.debug(f"✅ Document {doc_id} indexé: {result}")
                else:
                    logger.warning(f"⚠️ Indexation ambiguë pour {doc_id}: {result}")
                
                return success
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.index_document(doc_id, document)
            
            logger.error("❌ Aucun client disponible pour indexation")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation document {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document via le client approprié."""
        if not self._initialized or self._closed:
            return False
        
        if not doc_id:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                response = await asyncio.wait_for(
                    self.client.delete(index=self.index_name, id=doc_id),
                    timeout=10.0
                )
                return response.get("result") == "deleted"
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.delete_document(doc_id)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression document {doc_id}: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot via le client approprié."""
        if not self._initialized or self._closed or not documents:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                try:
                    from elasticsearch.helpers import async_bulk
                except ImportError:
                    logger.error("❌ elasticsearch.helpers non disponible")
                    return False
                
                actions = []
                for doc in documents:
                    if isinstance(doc, dict) and doc.get("id"):
                        action = {
                            "_index": self.index_name,
                            "_id": doc["id"],
                            "_source": doc
                        }
                        actions.append(action)
                
                if not actions:
                    logger.warning("⚠️ Aucune action valide pour bulk")
                    return False
                
                success, failed = await async_bulk(self.client, actions)
                total_docs = len(documents)
                success_count = total_docs - len(failed)
                
                logger.info(f"✅ Bulk indexation: {success_count}/{total_docs} succès")
                if failed:
                    logger.warning(f"⚠️ {len(failed)} échecs en bulk indexation")
                
                return len(failed) == 0
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.bulk_index(documents)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur bulk indexation: {e}")
            return False
    
    async def count_documents(self, user_id: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> int:
        """Compte le nombre de documents."""
        if not self._initialized or self._closed:
            return 0
        
        try:
            count_body = {}
            
            if user_id is not None or (filters and isinstance(filters, dict)):
                count_body["query"] = {"bool": {"must": []}}
                
                if user_id is not None:
                    count_body["query"]["bool"]["must"].append({"term": {"user_id": user_id}})
                
                if filters and isinstance(filters, dict):
                    for field, value in filters.items():
                        if value is not None and field:
                            count_body["query"]["bool"]["must"].append({"term": {field: value}})
            
            if self.client_type == 'elasticsearch' and self.client:
                response = await asyncio.wait_for(
                    self.client.count(index=self.index_name, body=count_body),
                    timeout=10.0
                )
                return int(response.get("count", 0))
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.count_documents(user_id, filters)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Erreur comptage documents: {e}")
            return 0
    
    async def refresh_index(self) -> bool:
        """Force le refresh de l'index pour rendre les documents visibles."""
        if not self._initialized or self._closed:
            return False
        
        try:
            if self.client_type == 'elasticsearch' and self.client:
                await asyncio.wait_for(
                    self.client.indices.refresh(index=self.index_name),
                    timeout=10.0
                )
                logger.debug(f"✅ Index {self.index_name} rafraîchi")
                return True
                
            elif self.client_type == 'bonsai' and self.bonsai_client:
                return await self.bonsai_client.refresh_index()
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur refresh index: {e}")
            return False
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Retourne les informations détaillées sur l'état du client."""
        base_info = {
            "initialized": self._initialized,
            "client_type": self.client_type,
            "elasticsearch_available": self.client is not None,
            "bonsai_available": self.bonsai_client is not None,
            "connection_attempts": self._connection_attempts,
            "last_health_check": self._last_health_check,
            "index_name": self.index_name,
            "closed": self._closed,
            "initialization_error": self._initialization_error
        }
        
        # Ajouter des infos spécifiques selon le client actif
        if self.client_type == 'elasticsearch' and self.client:
            try:
                # Essayer d'obtenir des infos sur le cluster
                cluster_info = await asyncio.wait_for(
                    self.client.info(), 
                    timeout=5.0
                )
                base_info["cluster_info"] = {
                    "name": cluster_info.get("cluster_name", "unknown"),
                    "version": cluster_info.get("version", {}).get("number", "unknown"),
                    "lucene_version": cluster_info.get("version", {}).get("lucene_version", "unknown")
                }
            except Exception as e:
                base_info["cluster_info"] = {"error": str(e)}
                
        elif self.client_type == 'bonsai' and self.bonsai_client:
            if hasattr(self.bonsai_client, 'get_client_info'):
                try:
                    bonsai_info = await self.bonsai_client.get_client_info()
                    base_info["bonsai_info"] = bonsai_info
                except Exception as e:
                    base_info["bonsai_info"] = {"error": str(e)}
        
        return base_info
    
    async def close(self):
        """Ferme proprement toutes les connexions."""
        if self._closed:
            logger.debug("🔒 Client déjà fermé")
            return
        
        logger.info("🔒 Fermeture du client Elasticsearch hybride...")
        self._closed = True
        self._initialized = False
        
        close_tasks = []
        
        # Fermer le client Elasticsearch s'il existe
        if self.client:
            logger.info("🔒 Fermeture client Elasticsearch officiel...")
            close_tasks.append(self._safe_close_elasticsearch())
            
        # Fermer le client Bonsai s'il existe
        if self.bonsai_client:
            logger.info("🔒 Fermeture client Bonsai HTTP...")
            try:
                if hasattr(self.bonsai_client, 'close'):
                    close_tasks.append(self.bonsai_client.close())
                else:
                    self.bonsai_client = None
            except Exception as e:
                logger.warning(f"⚠️ Erreur préparation fermeture Bonsai: {e}")
                self.bonsai_client = None
        
        # Attendre toutes les fermetures avec timeout global
        if close_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=15.0
                )
                logger.info("✅ Tous les clients fermés avec succès")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Timeout lors de la fermeture des clients")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture groupée: {e}")
        
        # Nettoyer les références
        self.client = None
        self.bonsai_client = None
        self.client_type = None
        
        logger.info("✅ Client hybride fermé")
    
    async def __aenter__(self):
        """Support pour l'utilisation avec 'async with'."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique lors de la sortie du contexte."""
        await self.close()
    
    def __repr__(self) -> str:
        """Représentation string du client pour debug."""
        status = "initialized" if self._initialized else "not_initialized"
        if self._closed:
            status = "closed"
        
        return (
            f"HybridElasticClient("
            f"type={self.client_type}, "
            f"status={status}, "
            f"attempts={self._connection_attempts}"
            f")"
        )
    
    # Méthodes utilitaires pour diagnostics
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test complet de la connexion avec diagnostic détaillé."""
        test_result = {
            "timestamp": time.time(),
            "client_type": self.client_type,
            "initialized": self._initialized,
            "closed": self._closed,
            "tests": {}
        }
        
        if not self._initialized or self._closed:
            test_result["tests"]["status"] = "client_not_ready"
            return test_result
        
        # Test de santé
        try:
            health_start = time.time()
            is_healthy = await self.is_healthy()
            health_time = time.time() - health_start
            
            test_result["tests"]["health_check"] = {
                "success": is_healthy,
                "time": health_time,
                "details": self._last_health_check
            }
        except Exception as e:
            test_result["tests"]["health_check"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test de comptage simple
        try:
            count_start = time.time()
            doc_count = await self.count_documents()
            count_time = time.time() - count_start
            
            test_result["tests"]["count_documents"] = {
                "success": True,
                "count": doc_count,
                "time": count_time
            }
        except Exception as e:
            test_result["tests"]["count_documents"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test de recherche simple
        try:
            search_start = time.time()
            search_results = await self.search(
                user_id=1, 
                query="test", 
                limit=1
            )
            search_time = time.time() - search_start
            
            test_result["tests"]["search"] = {
                "success": True,
                "results_count": len(search_results),
                "time": search_time
            }
        except Exception as e:
            test_result["tests"]["search"] = {
                "success": False,
                "error": str(e)
            }
        
        # Calculer le score global
        successful_tests = sum(1 for test in test_result["tests"].values() 
                             if isinstance(test, dict) and test.get("success", False))
        total_tests = len(test_result["tests"])
        
        test_result["overall_success"] = successful_tests == total_tests
        test_result["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return test_result
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Retourne un status détaillé pour monitoring et debugging."""
        status = {
            "client_info": {
                "class": self.__class__.__name__,
                "client_type": self.client_type,
                "initialized": self._initialized,
                "closed": self._closed,
                "connection_attempts": self._connection_attempts,
                "initialization_error": self._initialization_error
            },
            "health": {
                "last_check": self._last_health_check,
                "current_healthy": False
            },
            "configuration": {
                "index_name": self.index_name,
                "bonsai_url_configured": bool(settings.BONSAI_URL),
                "url_masked": self._mask_credentials(settings.BONSAI_URL) if settings.BONSAI_URL else None
            },
            "capabilities": {
                "search": False,
                "index": False,
                "delete": False,
                "bulk": False,
                "count": False
            }
        }
        
        # Tester la santé actuelle
        if self._initialized and not self._closed:
            try:
                status["health"]["current_healthy"] = await self.is_healthy()
                status["capabilities"] = {
                    "search": True,
                    "index": True,
                    "delete": True,
                    "bulk": True,
                    "count": True
                }
            except Exception as e:
                status["health"]["current_error"] = str(e)
        
        return status
    
    # Méthodes pour compatibilité et intégration
    
    def get_client_type(self) -> Optional[str]:
        """Retourne le type de client actuel."""
        return self.client_type
    
    def is_initialized(self) -> bool:
        """Vérifie si le client est initialisé."""
        return self._initialized and not self._closed
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Retourne les informations de connexion pour debug."""
        return {
            "client_type": self.client_type,
            "initialized": self._initialized,
            "closed": self._closed,
            "connection_attempts": self._connection_attempts,
            "has_elasticsearch_client": self.client is not None,
            "has_bonsai_client": self.bonsai_client is not None,
            "last_health_check": self._last_health_check,
            "initialization_error": self._initialization_error
        }