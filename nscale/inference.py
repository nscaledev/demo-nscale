"""
Inference service client for running predictions with deployed models.
"""

from typing import Dict, List, Optional
from .exceptions import NscaleError


class InferenceClient:
    """
    Client for inference operations using OpenAI-compatible API.

    Note: This uses a separate base URL (inference.api.nscale.com)
    and does not require organization ID in the path.
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = "https://inference.api.nscale.com/v1",
        timeout: int = 120,
    ):
        """
        Initialize inference client.

        Args:
            api_token: Nscale API token
            base_url: Base URL for inference API
            timeout: Request timeout in seconds
        """
        import requests

        if not api_token:
            raise NscaleError("API token is required")

        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set up session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        })

    def list_models(self) -> List[Dict]:
        """
        List available inference endpoints.

        Returns:
            List of available models with their metadata

        Raises:
            NscaleError: If request fails
        """
        url = f"{self.base_url}/models"
        response = self.session.get(url, timeout=self.timeout)

        if not response.ok:
            raise NscaleError(
                f"Failed to list models: {response.status_code} {response.text}"
            )

        result = response.json()
        return result.get("data", [])

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> Dict:
        """
        Generate a chat completion.

        Args:
            model: Model ID (e.g., "Qwen/Qwen3-4B-Instruct-2507")
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Response dictionary with completion

        Raises:
            NscaleError: If request fails
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        response = self.session.post(url, json=payload, timeout=self.timeout)

        if not response.ok:
            raise NscaleError(
                f"Chat completion failed: {response.status_code} {response.text}"
            )

        return response.json()

    def generate_answer(
        self,
        model: str,
        question: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate an answer to a question (convenience method).

        Args:
            model: Model ID
            question: The question to answer
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for chat_completion

        Returns:
            The generated answer as a string

        Raises:
            NscaleError: If request fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": question})

        response = self.chat_completion(model=model, messages=messages, **kwargs)

        # Extract content from response
        choices = response.get("choices", [])
        if not choices:
            raise NscaleError("No completion choices in response")

        return choices[0].get("message", {}).get("content", "")

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
