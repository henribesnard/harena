"""Main FastAPI application for Conversation Service V2."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from .api.v2.endpoints import conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Conversation Service V2...")
    yield
    # Shutdown
    logger.info("Shutting down Conversation Service V2...")


# Create FastAPI app
app = FastAPI(
    title="Harena Conversation Service V2",
    description="Assistant financier intelligent avec Text-to-SQL via DeepSeek API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Documentation accessible sur /docs
    redoc_url="/redoc",  # ReDoc accessible sur /redoc
    openapi_url="/openapi.json"  # Schema OpenAPI sur /openapi.json
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "https://harena.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Une erreur inattendue s'est produite",
                "details": str(exc) if app.debug else "Veuillez réessayer plus tard"
            },
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )


# Include routers
app.include_router(conversation.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Harena Conversation Service V2",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "docs_v2": "/api/v2/docs",
        "health": "/api/v2/health"
    }


# Add additional docs routes for /api/v2/* compatibility
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import HTMLResponse, JSONResponse


@app.get("/api/v2/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Swagger UI accessible également sur /api/v2/docs."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


@app.get("/api/v2/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc accessible également sur /api/v2/redoc."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


@app.get("/api/v2/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema accessible également sur /api/v2/openapi.json."""
    return JSONResponse(app.openapi())


if __name__ == "__main__":
    import uvicorn
    from .config.settings import settings
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
        log_level="info"
    )
