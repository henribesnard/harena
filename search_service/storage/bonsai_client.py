"""
Client Bonsai compatible - Alternative pour contourner les problèmes de compatibilité.
VERSION CORRIGÉE - Corrige le bug 'dict' object has no attribute 'lower'

Ce client utilise des requêtes HTTP directes pour interagir avec Bonsai
quand le client Elasticsearch officiel refuse la connexion.
"""
import logging
import time
import json
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
from urllib.parse import urlparse

from config_service.config import settings

logger = logging.getLogger("search_service.bonsai")
metrics_logger = logging.getLogger("search_service.metrics.bonsai")


class BonsaiClient:
    """Client HTTP direct pour Bonsai quand le client Elasticsearch ne fonctionne pas."""
    
    def __init__(self):
        self.base_url = None
        self.session = None
        self.index_name = "harena_transactions"
        self._initialized = False
        self._connection_attempts = 0
        self._last_health_check = None
        self.auth = None
        
    async def initialize(self):
        """Initialise la connexion Bonsai avec HTTP direct."""
        logger.info("🌐 Initialisation du client Bonsai HTTP...")
        
        if not settings.BONSAI_URL:
            logger.error("❌ BONSAI_URL non configurée")
            return False
        
        try:
            # Parser l'URL Bonsai pour extraire les informations
            parsed_url = urlparse(settings.BONSAI_URL)
            
            if not parsed_url.hostname:
                logger.error("❌ URL Bonsai malformée")
                return False
            
            # Construire l'URL de base
            self.base_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
            if parsed_url.port:
                self.base_url += f":{parsed_url.port}"
            
            # Extraire les credentials
            if parsed_url.username and parsed_url.password:
                self.auth = aiohttp.BasicAuth(parsed_url.username, parsed_url.password)
                logger.info(f"🔑 Authentification configurée pour {parsed_url.username}")
            
            # Masquer l'URL pour l'affichage
            safe_url = f"{parsed_url.scheme}://***:***@{parsed_url.hostname}"
            if parsed_url.port:
                safe_url += f":{parsed_url.port}"
            logger.info(f"🔗 Connexion Bonsai HTTP: {safe_url}")
            
            # Créer la session HTTP
            timeout = aiohttp.ClientTimeout(total=30.0)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30.0,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                auth=self.auth,
                timeout=timeout,
                connector=connector,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            # Test de connexion
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/") as response:
                connection_time = time.time() - start_time
                
                if response.status == 200:
                    cluster_info = await response.json()
                    logger.info(f"✅ Bonsai connecté en {connection_time:.2f}s")
                    logger.info(f"   Cluster: {cluster_info.get('cluster_name', 'Unknown')}")
                    logger.info(f"   Version: {cluster_info.get('version', {}).get('number', 'Unknown')}")
                else:
                    logger.error(f"❌ Échec connexion Bonsai: HTTP {response.status}")
                    return False
            
            # Test de santé du cluster
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    status = health.get("status", "red")
                    logger.info(f"💚 Santé cluster: {status}")
                    
                    if status in ["red"]:
                        logger.warning("⚠️ Cluster en état critique mais connexion établie")
                else:
                    logger.warning(f"⚠️ Impossible de vérifier la santé: HTTP {response.status}")
            
            # Test d'existence de l'index
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                if response.status == 200:
                    logger.info(f"✅ Index '{self.index_name}' trouvé")
                elif response.status == 404:
                    logger.warning(f"⚠️ Index '{self.index_name}' n'existe pas")
                else:
                    logger.warning(f"⚠️ Statut index inconnu: HTTP {response.status}")
            
            self._initialized = True
            logger.info("🎉 Client Bonsai HTTP initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Bonsai: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def is_healthy(self) -> bool:
        """Vérifie si le client est sain et fonctionnel."""
        if not self._initialized or not self.session:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    status = health.get("status", "red")
                    is_healthy = status in ["green", "yellow"]
                    
                    self._last_health_check = {
                        "timestamp": time.time(),
                        "healthy": is_healthy,
                        "status": status,
                        "client_type": "bonsai_http"
                    }
                    
                    return is_healthy
                return False
                
        except Exception as e:
            logger.error(f"❌ Health check Bonsai échoué: {e}")
            return False
    
    async def search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        filters: Dict[str, Any] = None,
        include_highlights: bool = True
    ) -> List[Dict[str, Any]]:
        """Recherche des transactions via Bonsai HTTP - VERSION CORRIGÉE."""
        
        if not self.session or not self._initialized:
            raise RuntimeError("Client Bonsai non initialisé")
        
        # VALIDATION CRITIQUE: S'assurer que query est une string
        if not isinstance(query, str):
            logger.error(f"❌ Query doit être une string dans BonsaiClient: {type(query)} = {query}")
            query = str(query) if query is not None else ""
        
        search_id = f"search_{int(time.time() * 1000)}"
        logger.info(f"🔍 [{search_id}] Recherche pour user {user_id}: '{query}' (limit: {limit})")
        
        start_time = time.time()
        
        try:
            # Expansion des termes de recherche - SÉCURISÉE
            from search_service.utils.query_expansion import expand_query_terms
            expanded_terms = expand_query_terms(query)
            
            # Validation des termes expandus
            validated_terms = []
            for term in expanded_terms:
                if isinstance(term, str):
                    validated_terms.append(term)
                else:
                    logger.warning(f"Term ignoré dans Bonsai: {type(term)} = {term}")
                    validated_terms.append(str(term))
            
            search_string = " ".join(validated_terms)
            
            # Construction de la requête de recherche
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}}
                        ],
                        "should": [
                            {
                                "multi_match": {
                                    "query": search_string,  # String validée
                                    "fields": ["searchable_text^3", "primary_description^2", "merchant_name^2", "category_name"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "terms": {
                                    "primary_description": validated_terms,  # Liste validée
                                    "boost": 2.0
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": limit,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"date": {"order": "desc"}}
                ]
            }
            
            # Log de debug
            logger.info(f"🔍 [{search_id}] Recherche via bonsai: search_string='{search_string}', terms={validated_terms}")
            
            # Ajouter les filtres si spécifiés
            if filters:
                search_body["query"]["bool"]["filter"] = []
                for field, value in filters.items():
                    if value is not None:
                        search_body["query"]["bool"]["filter"].append({"term": {field: value}})
            
            # Ajouter la mise en évidence si demandée
            if include_highlights:
                search_body["highlight"] = {
                    "fields": {
                        "searchable_text": {},
                        "primary_description": {},
                        "merchant_name": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            
            # Exécuter la recherche
            logger.info(f"🎯 [{search_id}] Exécution recherche lexicale...")
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                data=json.dumps(search_body),
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    logger.error(f"❌ Erreur recherche HTTP: {response.status}")
                    response_text = await response.text()
                    logger.error(f"   Response: {response_text}")
                    return []
                
                result = await response.json()
                query_time = time.time() - start_time
                
                # Analyser les résultats
                hits = result.get("hits", {}).get("hits", [])
                total_hits = result.get("hits", {}).get("total", {})
                
                if isinstance(total_hits, dict):
                    total_count = total_hits.get("value", 0)
                else:
                    total_count = total_hits
                
                # Formater les résultats
                results = []
                scores = []
                
                for hit in hits:
                    result_item = {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"]
                    }
                    
                    # Ajouter les highlights si disponibles
                    if "highlight" in hit:
                        result_item["highlights"] = hit["highlight"]
                    
                    results.append(result_item)
                    scores.append(hit["_score"])
                
                # Statistiques des scores
                if scores:
                    max_score = max(scores)
                    min_score = min(scores)
                    avg_score = sum(scores) / len(scores)
                else:
                    max_score = min_score = avg_score = 0
                
                # Logs de résultats
                logger.info(f"✅ [{search_id}] Recherche terminée en {query_time:.3f}s")
                logger.info(f"📊 [{search_id}] Résultats: {len(results)}/{total_count}")
                logger.info(f"🎯 [{search_id}] Scores: max={max_score:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
                
                # Métriques
                metrics_logger.info(
                    f"bonsai.search.success,"
                    f"user_id={user_id},"
                    f"query_time={query_time:.3f},"
                    f"results={len(results)},"
                    f"total={total_count},"
                    f"max_score={max_score:.3f}"
                )
                
                return results
                
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"❌ [{search_id}] Erreur recherche après {query_time:.3f}s: {e}")
            logger.error(f"   Query type: {type(query)}")
            logger.error(f"   Query value: {query}")
            metrics_logger.error(f"bonsai.search.failed,user_id={user_id},time={query_time:.3f},error={type(e).__name__}")
            return []
    
    async def index_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Indexe un document."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.put(
                f"{self.base_url}/{self.index_name}/_doc/{doc_id}",
                data=json.dumps(document),
                headers={"Content-Type": "application/json"}
            ) as response:
                return response.status in [200, 201]
                
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Supprime un document."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.delete(
                f"{self.base_url}/{self.index_name}/_doc/{doc_id}"
            ) as response:
                return response.status in [200, 404]  # 404 = déjà supprimé
                
        except Exception as e:
            logger.error(f"❌ Erreur suppression: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> bool:
        """Indexation en lot."""
        if not self.session or not self._initialized or not documents:
            return False
        
        try:
            # Construction du corps de la requête bulk
            bulk_body = []
            for doc in documents:
                # Action d'indexation
                action = {"index": {"_index": self.index_name, "_id": doc.get("id")}}
                bulk_body.append(json.dumps(action))
                bulk_body.append(json.dumps(doc))
            
            # Ajouter un retour à la ligne final
            bulk_data = "\n".join(bulk_body) + "\n"
            
            async with self.session.post(
                f"{self.base_url}/_bulk",
                data=bulk_data,
                headers={"Content-Type": "application/x-ndjson"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    errors = result.get("errors", False)
                    items = result.get("items", [])
                    
                    success_count = 0
                    error_count = 0
                    
                    for item in items:
                        if "index" in item:
                            if item["index"].get("status") in [200, 201]:
                                success_count += 1
                            else:
                                error_count += 1
                    
                    logger.info(f"✅ Bulk indexation: {success_count} succès, {error_count} erreurs")
                    return not errors
                else:
                    logger.error(f"❌ Erreur bulk HTTP: {response.status}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erreur bulk indexation: {e}")
            return False
    
    async def count_documents(self, user_id: int = None, filters: Dict[str, Any] = None) -> int:
        """Compte le nombre de documents."""
        if not self.session or not self._initialized:
            return 0
        
        try:
            count_body = {"query": {"match_all": {}}}
            
            if user_id is not None or filters:
                count_body["query"] = {"bool": {"must": []}}
                
                if user_id is not None:
                    count_body["query"]["bool"]["must"].append({"term": {"user_id": user_id}})
                
                if filters:
                    for field, value in filters.items():
                        if value is not None:
                            count_body["query"]["bool"]["must"].append({"term": {field: value}})
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_count",
                data=json.dumps(count_body),
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("count", 0)
                return 0
                
        except Exception as e:
            logger.error(f"❌ Erreur comptage documents: {e}")
            return 0
    
    async def refresh_index(self) -> bool:
        """Force le refresh de l'index pour rendre les documents visibles."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_refresh"
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Erreur refresh index: {e}")
            return False
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'index."""
        if not self.session or not self._initialized:
            return {"error": "Client non initialisé"}
        
        try:
            async with self.session.get(f"{self.base_url}/{self.index_name}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Erreur info index: {e}")
            return {"error": str(e)}
    
    async def create_index(self, mapping: Dict[str, Any] = None) -> bool:
        """Crée l'index avec un mapping optionnel."""
        if not self.session or not self._initialized:
            return False
        
        try:
            body = {}
            if mapping:
                body["mappings"] = mapping
            
            async with self.session.put(
                f"{self.base_url}/{self.index_name}",
                data=json.dumps(body) if body else None,
                headers={"Content-Type": "application/json"} if body else None
            ) as response:
                
                if response.status in [200, 201]:
                    logger.info(f"✅ Index '{self.index_name}' créé")
                    return True
                elif response.status == 400:
                    result = await response.json()
                    if "resource_already_exists_exception" in str(result):
                        logger.info(f"ℹ️ Index '{self.index_name}' existe déjà")
                        return True
                    else:
                        logger.error(f"❌ Erreur création index: {result}")
                        return False
                else:
                    logger.error(f"❌ Erreur création index: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur création index: {e}")
            return False
    
    async def delete_index(self) -> bool:
        """Supprime l'index."""
        if not self.session or not self._initialized:
            return False
        
        try:
            async with self.session.delete(f"{self.base_url}/{self.index_name}") as response:
                if response.status in [200, 404]:  # 404 = déjà supprimé
                    logger.info(f"✅ Index '{self.index_name}' supprimé")
                    return True
                else:
                    logger.error(f"❌ Erreur suppression index: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur suppression index: {e}")
            return False
    
    async def close(self):
        """Ferme la session HTTP."""
        if self.session:
            logger.info("🔒 Fermeture session Bonsai HTTP...")
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture session: {e}")
            finally:
                self.session = None
                self._initialized = False
                logger.info("✅ Session Bonsai fermée")
    
    def __del__(self):
        """Destructeur pour s'assurer que la session est fermée."""
        if self.session and not self.session.closed:
            logger.warning("⚠️ Session Bonsai non fermée explicitement")