"""
Custom exceptions for the Onboarding Assistant.
"""


class OnboardingAssistantError(Exception):
    """Base exception for all onboarding assistant errors."""
    pass


class RepositoryDownloadError(OnboardingAssistantError):
    """Raised when repository download/clone fails."""
    pass


class OnboardingError(OnboardingAssistantError):
    """Raised when repository onboarding fails."""
    pass


class StateTransitionError(OnboardingAssistantError):
    """Raised when an invalid state transition is attempted."""
    pass


class ParsingError(OnboardingAssistantError):
    """Raised when file parsing fails."""
    pass


class EmbeddingError(OnboardingAssistantError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(OnboardingAssistantError):
    """Raised when vector store operations fail."""
    pass