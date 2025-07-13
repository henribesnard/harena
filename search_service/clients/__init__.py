"""
🔌 Module Clients - Clients de services externes

Point d'entrée simplifié pour tous les clients de services externes
du Search Service. Expose les clients avec leurs fonctions d'initialisation.
"""

# === IMPORTS CLIENTS ===
from .base_client import (
   BaseClient,
   CircuitBreaker,
   ClientStatus,
   CircuitBreakerState,
   RetryConfig,
   CircuitBreakerConfig,
   HealthCheckConfig,
   ClientFactory
)

from .elasticsearch_client import (
   ElasticsearchClient,
   get_default_client,
   initialize_default_client,
   shutdown_default_client,
   test_elasticsearch_connection,
   quick_search,
   get_client_metrics
)

# === CLASSE GESTIONNAIRE SIMPLIFIÉE ===
class ClientManager:
   """
   Gestionnaire unifié pour tous les clients externes
   
   Centralise l'accès aux clients sans implémenter de logique.
   """
   def __init__(self):
       self.elasticsearch_client = None
       self.client_factory = ClientFactory()
   
   async def initialize(self):
       """Initialise tous les clients"""
       self.elasticsearch_client = await initialize_default_client()
   
   async def shutdown(self):
       """Arrête tous les clients"""
       await shutdown_default_client()
   
   def get_elasticsearch_client(self) -> ElasticsearchClient:
       """Retourne le client Elasticsearch"""
       return get_default_client()

# === INSTANCE GLOBALE ===
client_manager = ClientManager()

# === EXPORTS ===
__all__ = [
   # === GESTIONNAIRE PRINCIPAL ===
   "ClientManager",
   "client_manager",
   
   # === CLIENT DE BASE ===
   "BaseClient",
   "CircuitBreaker",
   "ClientStatus",
   "CircuitBreakerState",
   "RetryConfig",
   "CircuitBreakerConfig", 
   "HealthCheckConfig",
   "ClientFactory",
   
   # === CLIENT ELASTICSEARCH ===
   "ElasticsearchClient",
   "get_default_client",
   "initialize_default_client",
   "shutdown_default_client",
   "test_elasticsearch_connection",
   "quick_search",
   "get_client_metrics"
]