"""FastAPI application entrypoint for the conversation service.

This module creates the FastAPI application, applies common middleware and
includes the API routers.  It exposes an ``app`` object that can be used by
ASGI servers such as Uvicorn.
"""

import logging
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .api.exception_middleware import GlobalExceptionMiddleware
from .api.middleware import setup_middleware

app_state: Dict[str, Any] = {"health_status": "starting"}

from .api.routes import router as api_router, websocket_router
from .api.dependencies import cleanup_dependencies

from config_service.config import settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Load configuration from environment
    environment = settings.ENVIRONMENT
    cors_origins = settings.CORS_ORIGINS.split(",")
    allowed_hosts = ["localhost", "127.0.0.1"] + cors_origins

    # Basic OpenAI configuration check (replaces run_core_validation)
    if not settings.OPENAI_API_KEY or len(settings.OPENAI_API_KEY) < 10:
        logger.warning("OPENAI_API_KEY not configured - using mock responses")
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="Conversation Service MVP",
        description="""
        🤖 **AutoGen Multi-Agent Conversation Service**

        Sophisticated conversation AI powered by AutoGen v0.4 and OpenAI LLM,
        providing intelligent financial conversation processing with:
        
        - **Multi-Agent Architecture**: Specialized agents for intent detection, 
          entity extraction, query generation, and response synthesis
        - **Cost-Effective**: OpenAI models
        - **Real-time Processing**: Async conversation handling with context memory
        - **Production Ready**: Health monitoring, metrics, rate limiting
        
        **Architecture**: AutoGen RoundRobinGroupChat + OpenAI + Elasticsearch Integration
        """,
        version="1.0.0",
        contact={
            "name": "Harena Conversation Team",
            "email": "tech@harena.ai"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        },
        openapi_url=f"/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure trusted hosts (security)
    if environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    # Add custom middleware
    app.middleware("http")(add_process_time_header)
    app.middleware("http")(log_requests)
    app.add_middleware(GlobalExceptionMiddleware)
    
    # Apply shared middleware
    setup_middleware(app)

    # Include routes
    app.include_router(api_router)
    app.include_router(websocket_router)

    # Add root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "conversation_service_mvp",
            "version": "1.0.0",
            "status": app_state["health_status"],
            "documentation": "/docs",
            "health_check": "/health",
            "api_base": "/api/v1",
        }
    
    return app


async def validate_configuration() -> None:
    """
    Validate critical configuration settings from environment variables.
        
    Raises:
        ValueError: If critical configuration is missing or invalid
    """
    logger.info("🔧 Validating configuration")
    
    # Check OpenAI configuration
    if not openai_settings.OPENAI_API_KEY or len(openai_settings.OPENAI_API_KEY) < 10:
        logger.warning("⚠️ OPENAI_API_KEY not configured - using mock responses")

    # Check database configuration
    if not autogen_settings.DATABASE_URL:
        logger.warning("⚠️ DATABASE_URL not configured - using memory storage")

    # Check Redis configuration
    if not autogen_settings.REDIS_URL:
        logger.warning("⚠️ REDIS_URL not configured - using memory cache")

    # Validate port
    port = autogen_settings.PORT
    if not (1000 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    # Check environment-specific settings
    environment = autogen_settings.ENVIRONMENT
    debug = autogen_settings.DEBUG
    secret_key = autogen_settings.SECRET_KEY
    
    if environment == "production":
        if debug:
            logger.warning("⚠️ DEBUG enabled in production environment")
        
        if not secret_key or len(secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters in production")
    
    logger.info("✅ Configuration validation completed")


async def pre_initialize_dependencies() -> None:
    """
    Pre-initialize critical dependencies to catch startup errors early.

    Performs a full MVPTeamManager initialization and health check so that
    the service fails fast if any agent is unhealthy.
    """
    logger.info("🔄 Pre-initializing dependencies")

    try:
        # Test import of critical modules
        from .core import load_team_manager
        from .core.conversation_manager import ConversationManager
        from .core.metrics_collector import MetricsCollector

        MVPTeamManager, _ = load_team_manager()
        if MVPTeamManager is None:
            raise ImportError("MVPTeamManager not available")

        manager = MVPTeamManager()
        try:
            await manager.initialize_agents()
            logger.info("✅ MVPTeamManager health check passed")
        finally:
            await manager.shutdown()

        # Test configuration loading
        environment = autogen_settings.ENVIRONMENT
        logger.info(f"✅ Settings loaded for environment: {environment}")

    except ImportError as e:
        logger.error(f"❌ Failed to import core modules: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Dependency pre-initialization failed: {e}")
        raise


async def initialize_dependencies() -> None:
    """Initialize and validate core dependencies."""
    logger.info("🔧 Initializing dependencies")

    try:
        from .core import load_team_manager
        from .core.conversation_manager import ConversationManager

        # Instantiate core components to ensure availability
        MVPTeamManager, _ = load_team_manager()
        if MVPTeamManager is None:
            raise ImportError("MVPTeamManager not available")
        MVPTeamManager()
        conversation_manager = ConversationManager()
        await conversation_manager.initialize()

        logger.info("✅ Dependencies initialized successfully")
    except Exception as e:
        logger.error(f"❌ Dependency initialization failed: {e}")
        raise


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    try:
        await validate_configuration()
        await initialize_dependencies()
        app_state["health_status"] = "healthy"
        yield
    except Exception:
        app_state["health_status"] = "error"
        raise
    finally:
        await cleanup_dependencies()


# Middleware functions
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # Skip health check logging in production to reduce noise
    if request.url.path != "/health":
        logger.info(f"🔍 {request.method} {request.url.path} from {client_ip}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    if request.url.path != "/health":
        logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response


# Create application instance
app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("conversation_service.main:app", host="0.0.0.0", port=8000, reload=True)
