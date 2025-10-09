"""
Point d'entrée principal conversation service - Version réécrite compatible JWT
Correction des appels health_check synchrones
"""
import logging
import asyncio
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from dotenv import load_dotenv

# Charger le fichier .env au démarrage
load_dotenv()

# Configuration path pour imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Imports conversation service
from conversation_service.clients.deepseek_client import DeepSeekClient, DeepSeekError
from conversation_service.core.cache_manager import CacheManager
from conversation_service.api.routes.conversation import router as conversation_router
from conversation_service.api.middleware.auth_middleware import JWTAuthMiddleware
from conversation_service.utils.metrics_collector import metrics_collector
# Imports AutoGen supprimés - Phase 5 workflow intégré ne les utilise plus
# from conversation_service.autogen_core import ConversationServiceRuntime
# from conversation_service.teams import MultiAgentFinancialTeam
from config_service.config import settings

# Configuration logging optimisée
logging.basicConfig(
    level=getattr(logging, getattr(settings, 'CONVERSATION_SERVICE_LOG_LEVEL', 'INFO'), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("conversation_service")

class ConversationServiceLoader:
    """
    Service loader compatible user_service JWT avec validation renforcée
    """
    
    def __init__(self):
        self.deepseek_client: DeepSeekClient = None
        self.cache_manager: CacheManager = None
        # self.autogen_runtime: ConversationServiceRuntime = None  # Supprimé - Phase 5
        # self.multi_agent_team: MultiAgentFinancialTeam = None  # Supprimé - Phase 5
        self.service_healthy = False
        self.initialization_error = None
        self.service_start_time = datetime.now(timezone.utc)
        
        # Configuration service
        self.service_config = {
            "phase": 5,  # Phase 5 avec génération de réponses complètes
            "version": "2.0.0",  # Version Phase 5 avec workflow complet
            "features": ["intent_classification", "entity_extraction", "query_generation", "search_execution", "response_generation", "json_output", "cache", "auth", "metrics", "autogen_runtime", "contextual_responses", "insights_generation"],
            "json_output_enforced": True,
            "deepseek_model": getattr(settings, 'DEEPSEEK_CHAT_MODEL', 'deepseek-chat'),
            "jwt_compatible": True,
            "autogen_compatible": True,
        }
        
        logger.info("ConversationServiceLoader initialisé - Compatible JWT user_service")
    
    async def initialize_conversation_service(self, app: FastAPI) -> bool:
        """
        Initialise conversation service avec validation JWT compatible user_service
        """
        try:
            logger.info(" Initialisation Conversation Service - Compatible user_service JWT")
            
            # Configuration AutoGen - Définir OPENAI_API_KEY dummy pour éviter erreur initialisation
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = 'dummy-key-for-autogen-compatibility'
                logger.debug(" OPENAI_API_KEY dummy défini pour compatibilité AutoGen")
            
            # Vérification configuration service
            if not getattr(settings, 'CONVERSATION_SERVICE_ENABLED', True):
                logger.info(" Conversation Service désactivé par configuration")
                return False
            
            # Validation configuration JWT CRITIQUE
            jwt_validation = await self._validate_jwt_configuration()
            if not jwt_validation:
                logger.error("ERROR Configuration JWT incompatible avec user_service")
                return False
            
            # Validation configuration complète
            config_validation = await self._validate_comprehensive_configuration()
            if not config_validation:
                return False
            
            # Initialisation clients externes avec retry
            clients_success = await self._initialize_external_clients_with_retry()
            if not clients_success:
                return False
            
            # Validation JSON Output fonctionnelle
            json_validation = await self._validate_json_output_functionality()
            if not json_validation:
                logger.error("ERROR Validation JSON Output échouée")
                return False
            
            # Health check initial complet
            health_ok = await self._comprehensive_health_check()
            if not health_ok:
                logger.error("ERROR Health check initial échoué")
                return False
            
            # Test JWT end-to-end
            jwt_test_ok = await self._test_jwt_end_to_end()
            if not jwt_test_ok:
                logger.error("ERROR Test JWT end-to-end échoué")
                return False
            
            # Injection services dans app state
            self._inject_services_into_app_state(app)
            
            # Configuration middleware et routes APRÈS validation
            # NOTE: Routes déjà incluses au niveau module, mais middlewares CRITIQUES nécessaires
            self._configure_app_middleware_and_routes(app)
            
            # Phase 5 : Workflow complet intégré - Plus besoin d'AutoGen Runtime
            logger.info(" Phase 5 workflow intégré activé - AutoGen Runtime remplacé")
            
            # Warm-up optionnel du cache
            await self._optional_cache_warmup()
            
            # Finalisation
            self.service_healthy = True
            uptime = (datetime.now(timezone.utc) - self.service_start_time).total_seconds()
            
            logger.info(" Conversation Service initialisé avec succès")
            logger.info(f" Configuration: {len(self.service_config['features'])} fonctionnalités actives")
            logger.info(f" DeepSeek: {self.service_config['deepseek_model']} avec JSON Output forcé")
            logger.info(f" JWT: Compatible user_service")
            cache_status = "activé" if self.cache_manager else "désactivé"
            logger.info(f" Cache: Redis sémantique {cache_status}")
            logger.info(f"Temps initialisation: {uptime:.2f}s")
            
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.service_healthy = False
            logger.error(f"ERROR Erreur critique initialisation: {str(e)}", exc_info=True)
            return False
    
    async def _validate_jwt_configuration(self) -> bool:
        """Validation spécifique configuration JWT compatible user_service"""
        try:
            logger.info(" Validation configuration JWT user_service...")
            
            # Vérification SECRET_KEY
            secret_key = getattr(settings, 'SECRET_KEY', None)
            if not secret_key:
                logger.error("ERROR SECRET_KEY manquant - requis pour JWT")
                return False
            
            if len(secret_key) < 32:
                logger.error(f"ERROR SECRET_KEY trop court: {len(secret_key)} chars (minimum 32)")
                return False
            
            # Vérification algorithme JWT
            algorithm = getattr(settings, 'JWT_ALGORITHM', 'HS256')
            if algorithm not in ['HS256', 'HS384', 'HS512']:
                logger.error(f"ERROR Algorithme JWT non supporté: {algorithm}")
                return False
            
            logger.info(f" Configuration JWT validée - Algorithme: {algorithm}")
            logger.info(f" SECRET_KEY: {len(secret_key)} caractères")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR Erreur validation JWT: {str(e)}")
            return False
    
    async def _validate_comprehensive_configuration(self) -> bool:
        """Validation configuration complète avec diagnostic détaillé"""
        try:
            validation_errors = []
            validation_warnings = []
            
            # DeepSeek API Key
            if not getattr(settings, 'DEEPSEEK_API_KEY', None):
                validation_errors.append("DEEPSEEK_API_KEY manquant")
            else:
                api_key = settings.DEEPSEEK_API_KEY
                if len(api_key) < 20:
                    validation_errors.append("DEEPSEEK_API_KEY semble invalide (trop court)")
                if not api_key.startswith(('sk-', 'test-', 'dummy')):
                    validation_warnings.append("DEEPSEEK_API_KEY format inhabituel")
            
            # Configuration DeepSeek
            deepseek_url = getattr(settings, 'DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
            if not deepseek_url.startswith('https://'):
                validation_warnings.append("DEEPSEEK_BASE_URL n'utilise pas HTTPS")
            
            # Configuration Cache (optionnelle)
            redis_config = getattr(settings, 'REDIS_URL', None)
            if not redis_config:
                validation_warnings.append("REDIS_URL non configuré - cache désactivé")
            
            # Configuration métriques
            if not getattr(settings, 'METRICS_ENABLED', True):
                validation_warnings.append("Métriques désactivées")
            
            # Log résultats validation
            if validation_errors:
                logger.error(f"ERROR Erreurs configuration: {', '.join(validation_errors)}")
                return False
            
            if validation_warnings:
                for warning in validation_warnings:
                    logger.warning(f" Configuration: {warning}")
            
            logger.info(" Configuration générale validée")
            return True
            
        except Exception as e:
            logger.error(f"ERROR Erreur validation configuration: {str(e)}")
            return False
    
    async def _initialize_external_clients_with_retry(self) -> bool:
        """Initialisation clients externes avec retry intelligent"""
        try:
            # Initialisation DeepSeek client avec retry
            logger.info(" Initialisation client DeepSeek...")
            
            for attempt in range(3):
                try:
                    self.deepseek_client = DeepSeekClient()
                    await self.deepseek_client.initialize()
                    break
                except DeepSeekError as e:
                    logger.warning(f"Tentative {attempt + 1} DeepSeek échouée: {str(e)}")
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            #  CORRECTION: Test connexion DeepSeek - méthode synchrone
            deepseek_healthy = self.deepseek_client.health_check()  # Pas de await
            if not deepseek_healthy:
                logger.error("ERROR DeepSeek API non accessible")
                return False
            
            logger.debug("DeepSeek client ready")
            
            # Initialisation Cache Manager (non critique)
            logger.info(" Initialisation cache Redis...")
            try:
                self.cache_manager = CacheManager()
                await self.cache_manager.initialize()
                
                #  CORRECTION: Test connexion Redis - méthode synchrone ou gestion async
                try:
                    if hasattr(self.cache_manager, 'health_check'):
                        # Vérifier si la méthode est async ou sync
                        import inspect
                        if inspect.iscoroutinefunction(self.cache_manager.health_check):
                            cache_healthy = await self.cache_manager.health_check()
                        else:
                            cache_healthy = self.cache_manager.health_check()
                    else:
                        # Si pas de health_check, considérer comme sain
                        cache_healthy = True
                        
                    if cache_healthy:
                        logger.debug("Redis cache ready")
                    else:
                        logger.warning(" Redis indisponible - cache désactivé")
                        self.cache_manager = None
                except Exception as health_error:
                    logger.warning(f" Erreur health check Redis: {str(health_error)}")
                    # Garder le cache manager même si health check échoue
                        
            except Exception as e:
                logger.warning(f" Cache Redis non disponible: {str(e)}")
                self.cache_manager = None
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR Erreur initialisation clients: {str(e)}")
            return False
    
    async def _validate_json_output_functionality(self) -> bool:
        """Validation fonctionnelle JSON Output DeepSeek"""
        try:
            logger.info(" Validation JSON Output fonctionnalité...")
            
            if not self.deepseek_client:
                logger.error("ERROR DeepSeek client non disponible pour validation JSON")
                return False
            
            # Test JSON Output avec prompt simple
            test_response = await self.deepseek_client.chat_completion(
                messages=[{
                    "role": "system", 
                    "content": "Tu réponds uniquement en JSON valide."
                }, {
                    "role": "user", 
                    "content": "Teste JSON Output avec cette structure: {\"test\": true, \"message\": \"validation\"}"
                }],
                max_tokens=100,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Validation réponse
            if not test_response or "choices" not in test_response:
                logger.error("ERROR Test JSON Output: réponse invalide")
                return False
            
            content = test_response["choices"][0]["message"]["content"]
            
            # Validation JSON parsing
            import json
            try:
                parsed_json = json.loads(content)
                if not isinstance(parsed_json, dict):
                    logger.error("ERROR Test JSON Output: format non-objet")
                    return False
                    
                logger.info(f" JSON Output fonctionnel: {parsed_json}")
                return True
                
            except json.JSONDecodeError as e:
                logger.error(f"ERROR Test JSON Output: parsing échoué - {str(e)}")
                logger.error(f"Contenu reçu: {content}")
                return False
            
        except Exception as e:
            logger.error(f"ERROR Erreur validation JSON Output: {str(e)}")
            return False
    
    # Note: Les méthodes _initialize_autogen_runtime et _initialize_multi_agent_team
    # ont été supprimées car obsolètes avec le workflow Phase 5 intégré
    
    async def _comprehensive_health_check(self) -> bool:
        """Health check complet multi-services avec diagnostic"""
        try:
            logger.info(" Health check complet...")
            
            health_results = {
                "deepseek": {"status": False, "details": ""},
                "cache": {"status": False, "details": ""},
                "json_output": {"status": False, "details": ""},
                "configuration": {"status": True, "details": "OK"}
            }
            
            #  CORRECTION: Test DeepSeek - méthode synchrone
            if self.deepseek_client:
                try:
                    deepseek_health = self.deepseek_client.health_check()  # Pas de await
                    health_results["deepseek"]["status"] = deepseek_health
                    health_results["deepseek"]["details"] = "Opérationnel" if deepseek_health else "Inaccessible"
                except Exception as e:
                    health_results["deepseek"]["details"] = f"Erreur: {str(e)}"
            
            #  CORRECTION: Test Cache - gestion async/sync
            if self.cache_manager:
                try:
                    if hasattr(self.cache_manager, 'health_check'):
                        import inspect
                        if inspect.iscoroutinefunction(self.cache_manager.health_check):
                            cache_health = await self.cache_manager.health_check()
                        else:
                            cache_health = self.cache_manager.health_check()
                    else:
                        cache_health = True
                        
                    health_results["cache"]["status"] = cache_health
                    health_results["cache"]["details"] = "Opérationnel" if cache_health else "Indisponible"
                except Exception as e:
                    health_results["cache"]["details"] = f"Erreur: {str(e)}"
            else:
                health_results["cache"]["details"] = "Désactivé"
            
            # Validation JSON Output déjà effectuée
            health_results["json_output"]["status"] = True
            health_results["json_output"]["details"] = "Validé"
            
            # Évaluation globale
            critical_services = ["deepseek", "json_output", "configuration"]
            critical_ok = all(health_results[service]["status"] for service in critical_services)
            
            # Log détaillé
            for service, result in health_results.items():
                status_icon = "" if result["status"] else "ERROR"
                logger.info(f"{status_icon} {service.title()}: {result['details']}")
            
            if critical_ok:
                logger.debug("Health check: critical services operational")
            else:
                logger.error(" Health check global: ERROR Services critiques défaillants")
            
            return critical_ok
            
        except Exception as e:
            logger.error(f"ERROR Erreur health check: {str(e)}")
            return False
    
    async def _test_jwt_end_to_end(self) -> bool:
        """Test JWT end-to-end pour valider la compatibilité user_service"""
        try:
            logger.info(" Test JWT end-to-end user_service...")
            
            # Import dynamique pour éviter les dépendances circulaires
            try:
                from user_service.core.security import create_access_token
                from conversation_service.api.middleware.auth_middleware import JWTValidator
            except ImportError as e:
                logger.warning(f" Cannot import user_service modules: {e}")
                # Mode dégradé - test basique uniquement
                return await self._test_jwt_basic()
            
            # Test avec un token généré par user_service
            test_token = create_access_token(subject=42)
            
            # Validation par conversation_service
            validator = JWTValidator()
            result = validator.validate_token(test_token)
            
            if result.success:
                logger.info(f" JWT end-to-end OK - User ID: {result.user_id}")
                logger.info(f" Payload validé: {list(result.token_payload.keys())}")
                return True
            else:
                logger.error(f"ERROR JWT end-to-end échoué: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"ERROR Erreur test JWT end-to-end: {str(e)}")
            return False
    
    async def _test_jwt_basic(self) -> bool:
        """Test JWT basique en cas d'impossibilité de test end-to-end"""
        try:
            logger.info(" Test JWT basique (mode dégradé)...")
            
            from conversation_service.api.middleware.auth_middleware import JWTValidator
            from jose import jwt
            import time
            
            # Créer un token test basique
            payload = {
                "sub": "42",
                "exp": int(time.time()) + 3600,
                "permissions": ["chat:write"]
            }
            
            test_token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
            
            # Validation
            validator = JWTValidator()
            result = validator.validate_token(test_token)
            
            if result.success:
                logger.info(" JWT basique OK")
                return True
            else:
                logger.error(f"ERROR JWT basique échoué: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"ERROR Erreur test JWT basique: {str(e)}")
            return False
    
    def _inject_services_into_app_state(self, app: FastAPI) -> None:
        """Injection services dans app state"""
        # Services principaux
        app.state.conversation_service = self
        app.state.deepseek_client = self.deepseek_client
        app.state.cache_manager = self.cache_manager
        # app.state.autogen_runtime = self.autogen_runtime  # Supprimé - Phase 5
        # app.state.multi_agent_team = self.multi_agent_team  # Supprimé - Phase 5
        
        # Configuration service
        app.state.service_config = self.service_config
        app.state.service_start_time = self.service_start_time
        
        # Métriques globales
        app.state.metrics_collector = metrics_collector
        
        # Metadata pour debugging
        app.state.service_metadata = {
            "initialization_time": datetime.now(timezone.utc),
            "python_version": sys.version,
            "service_loader_version": "1.1.0",
            "jwt_compatible": True
        }
        
        logger.info(" Services injectés dans app state")
    
    def _configure_app_middleware_and_routes(self, app: FastAPI) -> None:
        """Configuration middleware et routes - NOP car déjà fait au niveau module"""
        # NOTE: Middlewares CORS/JWT et routes santé déjà configurés aux lignes 743-818
        # Cette méthode est conservée pour compatibilité mais ne fait plus rien
        logger.info(" Middlewares et routes deja configures au niveau module")
        pass
    
    async def _optional_cache_warmup(self) -> None:
        """Warm-up optionnel du cache avec données communes"""
        if not self.cache_manager:
            return
        
        try:
            # Données pour warm-up
            warmup_data = [
                {
                    "key": "common_greeting",
                    "data": {"intent": "GREETING", "confidence": 0.99},
                    "type": "intent"
                },
                {
                    "key": "common_balance",
                    "data": {"intent": "BALANCE_INQUIRY", "confidence": 0.98},
                    "type": "intent"
                }
            ]
            
            # Vérifier si la méthode warm_up_cache existe
            if hasattr(self.cache_manager, 'warm_up_cache'):
                warmed = await self.cache_manager.warm_up_cache(warmup_data)
                if warmed > 0:
                    logger.info(f" Cache warm-up: {warmed} entrées préchargées")
            else:
                logger.debug("Cache warm-up non supporté par ce cache manager")
                
        except Exception as e:
            logger.debug(f"Cache warm-up optionnel échoué: {str(e)}")
    
    async def cleanup(self) -> None:
        """Nettoyage ressources avec métriques finales"""
        try:
            cleanup_start = datetime.now(timezone.utc)
            
            # Métriques finales
            if self.service_healthy:
                uptime = (cleanup_start - self.service_start_time).total_seconds()
                final_metrics = metrics_collector.get_all_metrics()
                
                logger.info(f" Métriques finales - Uptime: {uptime:.1f}s")
                logger.info(f" Requêtes totales: {final_metrics.get('counters', {}).get('conversation.requests.total', 0)}")
                logger.info(f" Taux succès: {100 - (final_metrics.get('counters', {}).get('conversation.errors.technical', 0) / max(final_metrics.get('counters', {}).get('conversation.requests.total', 1), 1) * 100):.1f}%")
            
            # Fermeture clients
            if self.deepseek_client:
                await self.deepseek_client.close()
                logger.info(" DeepSeek client fermé")
            
            if self.cache_manager:
                await self.cache_manager.close()
                logger.info(" Cache manager fermé")
            
            # if self.autogen_runtime:  # Supprimé - Phase 5
            #     await self.autogen_runtime.shutdown()
            #     logger.info(" AutoGen Runtime arrêté")
            
            # if self.multi_agent_team:  # Supprimé - Phase 5 
            #     logger.info(" Équipe multi-agents déchargée")
            
            cleanup_time = (datetime.now(timezone.utc) - cleanup_start).total_seconds()
            logger.info(f" Nettoyage terminé en {cleanup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"ERROR Erreur nettoyage: {str(e)}")

# Instance globale service loader
conversation_service_loader = ConversationServiceLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire cycle de vie application avec validation JWT"""
    startup_start = datetime.now(timezone.utc)

    # Startup
    logger.info(" Démarrage application conversation service - JWT compatible")

    try:
        # Initialisation du nouveau système v2.0 (app_state)
        from conversation_service.api.dependencies import app_state
        logger.info(" Initialisation pipeline v2.0...")

        try:
            v2_initialization = await asyncio.wait_for(
                app_state.initialize(),
                timeout=60.0
            )
            if v2_initialization:
                logger.info(" Pipeline v2.0 initialisé avec succès")
            else:
                logger.warning(" Pipeline v2.0 non initialisé - fallback sur legacy")
        except Exception as e:
            logger.warning(f" Erreur initialisation v2.0: {e} - fallback sur legacy")

        # Initialisation service legacy avec timeout
        initialization_success = await asyncio.wait_for(
            conversation_service_loader.initialize_conversation_service(app),
            timeout=90.0  # 90s timeout pour permettre les tests JWT
        )

        startup_time = (datetime.now(timezone.utc) - startup_start).total_seconds()

        if initialization_success:
            logger.info(f" Service démarré avec succès en {startup_time:.2f}s")
        else:
            logger.error(f"ERROR Échec initialisation en {startup_time:.2f}s - service dégradé")
            # App démarre quand même pour exposer health checks

        yield  # Application running
        
    except asyncio.TimeoutError:
        logger.error("ERROR Timeout initialisation service (90s)")
        yield  # App démarre en mode dégradé
        
    except Exception as e:
        logger.error(f"ERROR Erreur critique startup: {str(e)}", exc_info=True)
        yield  # App démarre en mode dégradé
    
    finally:
        # Shutdown
        shutdown_start = datetime.now(timezone.utc)
        logger.info(" Arrêt application conversation service")

        try:
            # Fermer app_state v2.0 d'abord
            from conversation_service.api.dependencies import app_state
            if app_state.initialized:
                await app_state.close()
                logger.info(" Pipeline v2.0 fermé")

            # Puis fermer le legacy
            await conversation_service_loader.cleanup()
            shutdown_time = (datetime.now(timezone.utc) - shutdown_start).total_seconds()
            logger.info(f" Arrêt propre terminé en {shutdown_time:.2f}s")
        except Exception as e:
            logger.error(f"ERROR Erreur arrêt: {str(e)}")

# Application FastAPI avec configuration JWT compatible
app = FastAPI(
    title="Harena Conversation Service",
    description="Service IA conversationnelle financière - Compatible user_service JWT",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",  # Docs activées même en production (pour debug)
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# IMPORTANT: Configuration CORS et JWT AVANT les routes
# CORS doit être ajouté en premier pour être exécuté en dernier (ordre inverse FastAPI)
environment = getattr(settings, 'ENVIRONMENT', 'production')
if environment in ["development", "testing"]:
    cors_origins = ["*"]
else:
    cors_origins = [
        "https://app.harena.fr",
        "https://api.harena.fr",
        "https://harenabackend-ab1b255e55c6.herokuapp.com"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)
logger.info(f"CORS configured - {len(cors_origins)} origins allowed")

# JWT middleware (exécuté avant CORS car ajouté après)
app.add_middleware(JWTAuthMiddleware)
logger.info("JWT middleware configured - user_service compatible")

# IMPORTANT: Inclure les routes APRÈS les middlewares
# Routes conversation v1 - le router a déjà son préfixe /api/v1/conversation
app.include_router(conversation_router)
logger.info("Routes conversation v1 configurees")

# Routes conversation v2.0 (nouvelle architecture)
try:
    from conversation_service.api.routes.conversation_v2 import router as conversation_v2_router
    app.include_router(conversation_v2_router)
    logger.info("Routes conversation v2.0 configurees")
except ImportError as e:
    logger.warning(f"Routes v2.0 non disponibles: {e}")

# Routes de santé au niveau module (avant lifespan)
@app.get("/health/live")
async def liveness_probe():
    """Probe liveness pour Kubernetes/Docker"""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/health/ready")
async def readiness_probe():
    """Probe readiness pour Kubernetes/Docker"""
    try:
        deepseek_ready = (
            conversation_service_loader.deepseek_client and
            conversation_service_loader.deepseek_client.health_check()
        )
    except Exception as e:
        logger.warning(f"Erreur readiness DeepSeek: {str(e)}")
        deepseek_ready = False

    is_ready = (
        conversation_service_loader.service_healthy and
        conversation_service_loader.deepseek_client and
        deepseek_ready
    )

    status_code = 200 if is_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "service_healthy": conversation_service_loader.service_healthy,
            "jwt_compatible": conversation_service_loader.service_config.get("jwt_compatible", False) if conversation_service_loader.service_config else False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Health check global principal (compatible pattern Harena)
@app.get("/health")
async def global_health():
    """Health check global avec statut JWT"""
    try:
        if conversation_service_loader.service_healthy:
            health_metrics = metrics_collector.get_health_metrics()
            service_uptime = (datetime.now(timezone.utc) - conversation_service_loader.service_start_time).total_seconds()
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "service": "conversation_service",
                    "phase": conversation_service_loader.service_config["phase"],
                    "version": conversation_service_loader.service_config["version"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": service_uptime,
                    "jwt_compatible": conversation_service_loader.service_config["jwt_compatible"],
                    "health_details": {
                        "total_requests": health_metrics.get("total_requests", 0),
                        "error_rate_percent": health_metrics.get("error_rate_percent", 0),
                        "avg_latency_ms": health_metrics.get("latency_p95_ms", 0)
                    },
                    "components": {
                        "deepseek_api": "operational",
                        "redis_cache": "operational" if conversation_service_loader.cache_manager else "disabled",
                        "intent_classification": "operational",
                        "json_output": "enforced",
                        "jwt_auth": "active_compatible",
                        "autogen_runtime": "deprecated_phase5_integrated"
                    },
                    "features": conversation_service_loader.service_config["features"]
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": "conversation_service",
                    "phase": 1,
                    "error": conversation_service_loader.initialization_error or "Service non initialisé",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "jwt_compatible": False,
                    "components": {
                        "deepseek_api": "unknown",
                        "redis_cache": "unknown", 
                        "intent_classification": "unavailable",
                        "json_output": "unknown",
                        "jwt_auth": "unknown",
                        "autogen_runtime": "unknown"
                    }
                }
            )
            
    except Exception as e:
        logger.error(f"ERROR Erreur health check global: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "service": "conversation_service", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "jwt_compatible": False
            }
        )

# Endpoint métriques compatible Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    """Métriques Prometheus pour monitoring"""
    try:
        if not conversation_service_loader.service_healthy:
            raise HTTPException(status_code=503, detail="Service non opérationnel")
        
        metrics_data = metrics_collector.get_all_metrics()
        service_uptime = (datetime.now(timezone.utc) - conversation_service_loader.service_start_time).total_seconds()
        
        # Format compatible monitoring Harena
        return {
            "service": "conversation_service",
            "timestamp": metrics_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "uptime_seconds": service_uptime,
            "metrics": metrics_data,
            "service_info": {
                "phase": conversation_service_loader.service_config["phase"],
                "version": conversation_service_loader.service_config["version"],
                "features": conversation_service_loader.service_config["features"],
                "json_output_enforced": conversation_service_loader.service_config["json_output_enforced"],
                "jwt_compatible": conversation_service_loader.service_config["jwt_compatible"],
                "autogen_compatible": conversation_service_loader.service_config["autogen_compatible"],
                "autogen_status": {"available": False, "status": "deprecated_phase5_integrated"}
            },
            "labels": {
                "service": "conversation_service",
                "phase": str(conversation_service_loader.service_config["phase"]),
                "version": conversation_service_loader.service_config["version"],
                "environment": getattr(settings, 'ENVIRONMENT', 'production'),
                "jwt_compatible": "true"
            }
        }
        
    except Exception as e:
        logger.error(f"ERROR Erreur export métriques: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur métriques")

# Handler erreurs globales optimisé
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler erreurs globales avec logging détaillé"""
    error_id = f"err_{int(datetime.now(timezone.utc).timestamp())}"
    
    logger.error(
        f"ERROR [{error_id}] Erreur non gérée: {exc.__class__.__name__}: {str(exc)} | "
        f"Path: {getattr(request, 'url', {}).path if hasattr(request, 'url') else 'unknown'} | "
        f"Method: {getattr(request, 'method', 'unknown')}",
        exc_info=True
    )
    
    metrics_collector.increment_counter("conversation.errors.unhandled")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du service",
            "service": "conversation_service",
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_path": str(request.url.path) if hasattr(request, 'url') else "unknown"
        }
    )

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info(" Démarrage direct conversation service - JWT compatible")
    
    # Configuration uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": getattr(settings, 'CONVERSATION_SERVICE_HOST', '0.0.0.0'),
        "port": getattr(settings, 'CONVERSATION_SERVICE_PORT', 8001),
        "reload": getattr(settings, 'CONVERSATION_SERVICE_DEBUG', False),
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
        "server_header": False,  # Sécurité
        "date_header": False,    # Sécurité
    }
    
    logger.info(f" Configuration: {uvicorn_config}")
    uvicorn.run(**uvicorn_config)