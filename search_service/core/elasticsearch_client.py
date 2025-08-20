import logging
import aiohttp
import ssl
import threading
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

# Instance globale
_default_client: Optional['ElasticsearchClient'] = None
_client_lock = threading.Lock()

class ElasticsearchClient:
    """
    Client Elasticsearch unifié pour le search service
    Basé sur BONSAI_URL et compatible avec l'architecture existante
    """
    
    def __init__(self):
        self.base_url = self._resolve_bonsai_url()
        self.index_name = settings.ELASTICSEARCH_INDEX
        self.session: Optional[aiohttp.ClientSession] = None
        self.ssl_context = ssl.create_default_context()
        
        # Headers pour Elasticsearch
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        logger.info(f"✅ ElasticsearchClient initialized")
        logger.info(f"🔗 URL: {self.base_url}")
        logger.info(f"📋 Index: {self.index_name}")
    
    def _resolve_bonsai_url(self) -> str:
        """Résout l'URL Bonsai depuis la configuration"""
        if settings.BONSAI_URL and settings.BONSAI_URL.strip():
            url = settings.BONSAI_URL.strip()
            logger.info(f"🔗 Using BONSAI_URL: {url}")
            return url
        else:
            raise RuntimeError(
                "❌ BONSAI_URL not configured. Please set BONSAI_URL in your .env file.\n"
                "Example: BONSAI_URL=https://your-cluster.eu-west-1.bonsaisearch.net:443"
            )
    
    async def initialize(self):
        """Initialise la session HTTP"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(
                ssl=self.ssl_context,
                limit=20,
                keepalive_timeout=30
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            )
            
            logger.info("🚀 HTTP session initialized")
            
            # Test de connectivité
            try:
                await self._test_connection()
            except Exception as e:
                logger.warning(f"⚠️ Initial connection test failed: {e}")
    
    async def _test_connection(self):
        """Test de connectivité basique"""
        async with self.session.get(self.base_url) as response:
            if response.status == 200:
                cluster_info = await response.json()
                logger.info(f"✅ Connected to Elasticsearch cluster: {cluster_info.get('cluster_name', 'unknown')}")
            else:
                logger.warning(f"⚠️ Connection test returned status {response.status}")
    
    async def search(self, index: str, body: Dict[str, Any], size: int = 20, from_: int = 0) -> Dict[str, Any]:
        """
        Exécute une recherche Elasticsearch
        Interface compatible avec le SearchEngine
        """
        if not self.session:
            await self.initialize()
        
        # Construction de l'URL de recherche
        search_url = f"{self.base_url}/{index}/_search"
        
        # Ajout des paramètres de pagination au body
        body["size"] = size
        body["from"] = from_
        
        try:
            async with self.session.post(search_url, json=body) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Search completed: {result.get('hits', {}).get('total', {}).get('value', 0)} hits")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Search failed with status {response.status}: {error_text}")
                    raise RuntimeError(f"Elasticsearch search failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during search: {e}")
            raise RuntimeError(f"Network error during search: {e}")
    
    async def count(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compte les documents correspondant à une requête
        """
        if not self.session:
            await self.initialize()
        
        count_url = f"{self.base_url}/{index}/_count"
        
        try:
            async with self.session.post(count_url, json=body) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Count completed: {result.get('count', 0)} documents")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Count failed with status {response.status}: {error_text}")
                    return {"count": 0}
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during count: {e}")
            return {"count": 0}
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du cluster"""
        if not self.session:
            await self.initialize()
        
        try:
            # Test de connectivité simple
            async with self.session.get(f"{self.base_url}/_cluster/health") as response:
                if response.status == 200:
                    health = await response.json()
                    return {
                        "status": "healthy",
                        "cluster_status": health.get("status", "unknown"),
                        "number_of_nodes": health.get("number_of_nodes", 0)
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def close(self):
        """Ferme la session HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("✅ HTTP session closed")

# === FONCTIONS GLOBALES ===

async def get_default_client() -> ElasticsearchClient:
    """
    Retourne le client Elasticsearch par défaut
    Initialise le client si nécessaire (thread-safe)
    """
    global _default_client
    
    if _default_client is None:
        with _client_lock:
            if _default_client is None:  # Double-check locking
                _default_client = ElasticsearchClient()
                await _default_client.initialize()
                logger.info("✅ Default Elasticsearch client initialized")
    
    return _default_client

async def initialize_default_client() -> ElasticsearchClient:
    """
    Force l'initialisation du client par défaut
    """
    global _default_client
    
    with _client_lock:
        _default_client = ElasticsearchClient()
        await _default_client.initialize()
        logger.info("✅ Default Elasticsearch client force-initialized")
    
    return _default_client

async def shutdown_default_client():
    """
    Ferme proprement le client par défaut
    """
    global _default_client
    
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
        logger.info("✅ Default Elasticsearch client shutdown")