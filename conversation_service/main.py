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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from .api.middleware import GlobalExceptionMiddleware
from fastapi import FastAPI

from .api.routes import router as api_router
from .api.websocket import router as websocket_router
from .api.middleware import setup_middleware


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Load configuration from environment
    environment = os.getenv("ENVIRONMENT", "development")
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    allowed_hosts = ["localhost", "127.0.0.1"] + cors_origins

    # Validate core setup after environment configuration
    run_core_validation()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="Conversation Service MVP",
        description="""
        🤖 **AutoGen Multi-Agent Conversation Service**
        
        Sophisticated conversation AI powered by AutoGen v0.4 and DeepSeek LLM,
        providing intelligent financial conversation processing with:
        
        - **Multi-Agent Architecture**: Specialized agents for intent detection, 
          entity extraction, query generation, and response synthesis
        - **Cost-Effective**: DeepSeek LLM with 90% cost savings vs GPT-4
        - **Real-time Processing**: Async conversation handling with context memory
        - **Production Ready**: Health monitoring, metrics, rate limiting
        
        **Architecture**: AutoGen RoundRobinGroupChat + DeepSeek + Elasticsearch Integration
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
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Agent-Used"]
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
    
    # Include API routes
    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["conversation"]
    )
    
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
            "api_base": "/api/v1"
        }
    
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Conversation Service")
    setup_middleware(app)
    app.include_router(api_router)
    app.include_router(websocket_router)
    return app


async def validate_configuration() -> None:
    """
    Validate critical configuration settings from environment variables.
        
    Raises:
        ValueError: If critical configuration is missing or invalid
    """
    logger.info("🔧 Validating configuration")
    
    # Check DeepSeek configuration
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key or len(deepseek_key) < 10:
        logger.warning("⚠️ DEEPSEEK_API_KEY not configured - using mock responses")
    
    # Check database configuration
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("⚠️ DATABASE_URL not configured - using memory storage")
    
    # Check Redis configuration
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.warning("⚠️ REDIS_URL not configured - using memory cache")
    
    # Validate port
    port = int(os.getenv("PORT", "8000"))
    if not (1000 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")
    
    # Check environment-specific settings
    environment = os.getenv("ENVIRONMENT", "development")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    secret_key = os.getenv("SECRET_KEY", "")
    
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
        environment = os.getenv("ENVIRONMENT", "development")
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
