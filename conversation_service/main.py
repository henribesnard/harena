"""FastAPI application entrypoint for the conversation service.

This module creates the FastAPI application, applies common middleware and
includes the API routers.  It exposes an ``app`` object that can be used by
ASGI servers such as Uvicorn.
"""

from fastapi import FastAPI

from .api.routes import router as api_router
from .api.websocket import router as websocket_router
from .api.middleware import setup_middleware


def create_app() -> FastAPI:
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


# Exception handlers
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with structured logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation exception handler."""
    logger.warning(f"Validation error: {exc} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "status_code": 422,
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc} - {request.method} {request.url}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error",
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


# Create application instance
app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("conversation_service.main:app", host="0.0.0.0", port=8000, reload=True)
