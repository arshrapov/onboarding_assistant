"""
LLM Router - handles interactions with different LLM providers.
"""

from typing import Optional
import google.generativeai as genai
from app.config import settings


class LLMRouter:
    """Router for managing LLM provider interactions."""

    def __init__(self):
        """Initialize the LLM router with configured provider."""
        self._configure_gemini()

    def _configure_gemini(self) -> None:
        """Configure Google Gemini API."""
        genai.configure(api_key=settings.gemini_api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate text using Gemini API.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (default from settings)
            temperature: Temperature for generation (default from settings)
            model: Model name to use (default from settings)

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature or settings.temperature
        model = model or settings.gemini_model

        try:
            llm_model = genai.GenerativeModel(model)

            response = llm_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )

            return response.text

        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
