"""
Main application entry point for the Onboarding Assistant.
Initializes FastAPI application with onboarding workflow endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.onboarding_routes import router as onboarding_router
from app.config import settings


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Onboarding Assistant API",
        description="AI-powered repository onboarding with RAG (Retrieval-Augmented Generation)",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(onboarding_router)

    return app


# Create application instance
app = create_app()


@app.get("/", tags=["health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "running",
        "service": "Onboarding Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "onboarding": "/api/v1/onboarding",
            "health": "/health"
        }
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "repos_dir": settings.repos_dir,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info"
    )