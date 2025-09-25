from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from app.api import router as api_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Slay Canvas - Backend",
        description="AI-powered collaborative multimedia platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "message": "Slay Canvas Backend is running",
            "environment": settings.ENVIRONMENT
        }
    
    # OAuth configuration check (for debugging)
    @app.get("/debug/oauth-config")
    async def oauth_config_debug():
        return {
            "google_client_id_configured": bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_ID != "your-google-client-id"),
            "google_client_secret_configured": bool(settings.GOOGLE_CLIENT_SECRET and settings.GOOGLE_CLIENT_SECRET != "your-google-client-secret"),
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "frontend_origin": settings.FRONTEND_ORIGIN
        }
    
    return app


app = create_app()
