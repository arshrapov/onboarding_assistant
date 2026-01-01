"""API module for FastAPI endpoints."""

from app.api.onboarding_routes import router as onboarding_router

__all__ = ["onboarding_router"]