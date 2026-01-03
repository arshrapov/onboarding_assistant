"""
Main application entry point for the Onboarding Assistant.
Initializes FastAPI application with onboarding workflow endpoints and Gradio UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from app.api.onboarding_routes import router as onboarding_router
from app.ui.gradio_app import create_gradio_interface
from app.config import settings


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with Gradio UI.

    Returns:
        Configured FastAPI application instance with mounted Gradio UI
    """
    app = FastAPI(
        title="Onboarding Assistant API",
        description="AI-powered repository onboarding with RAG (Retrieval-Augmented Generation)",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(onboarding_router)

    # Create and mount Gradio interface at root path
    gradio_app = create_gradio_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    return app


# Create application instance
app = create_app()


@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Onboarding Assistant",
        "version": "1.0.0",
        "repos_dir": settings.repos_dir,
        "ui": "/",
        "api_docs": "/api/docs",
        "endpoints": {
            "ui": "/",
            "api": "/api/v1/onboarding",
            "health": "/api/health"
        }
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