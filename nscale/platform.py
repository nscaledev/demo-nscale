"""
Core client for Nscale API with authentication, error handling, and retry logic.
"""

import time
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    AuthenticationError,
    ValidationError,
    ResourceNotFoundError,
    RateLimitError,
    ServerError,
    APIError,
)


class NscaleClient:
    """
    Base client for Nscale API.

    Handles:
    - Authentication
    - Request/response formatting
    - Error handling and custom exceptions
    - Retry logic for transient failures
    """

    def __init__(
        self,
        api_token: str,
        organization_id: str,
        base_url: str,
        timeout: int = 120,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ):
        """
        Initialize the Nscale client.

        Args:
            api_token: Nscale API token
            organization_id: Organization ID
            base_url: Base URL for the API (e.g., https://fine-tuning.api.nscale.com/api/v1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for exponential retry delays
        """
        if not api_token:
            raise AuthenticationError("API token is required")
        if not organization_id:
            raise ValidationError("Organization ID is required")

        self.api_token = api_token
        self.organization_id = organization_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Build full base URL with organization
        self.org_base_url = f"{self.base_url}/organizations/{organization_id}"

        # Set up session with retry logic
        self.session = requests.Session()

        # Configure retry strategy for network/transient errors
        # Note: 500 errors are typically NOT retried as they indicate server bugs
        # We only retry on network errors and 502/503/504 (temporary server issues)
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[502, 503, 504],  # Retry on temporary server issues
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            raise_on_status=False,  # We handle status codes manually
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Default headers
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        })

    def _handle_error_response(self, response: requests.Response) -> None:
        """
        Handle error responses and raise appropriate exceptions.

        Args:
            response: The response object

        Raises:
            Various NscaleError subclasses based on status code
        """
        status_code = response.status_code
        body = response.text

        try:
            error_data = response.json()
            error_code = error_data.get("code", "unknown")
            error_message = error_data.get("message") or error_data.get("error", "Unknown error")
        except Exception:
            error_code = "unknown"
            error_message = body or "Unknown error"

        # Map status codes to exceptions
        if status_code == 401 or status_code == 403:
            raise AuthenticationError(f"Authentication failed: {error_message}")
        elif status_code == 404:
            raise ResourceNotFoundError(f"Resource not found: {error_message}")
        elif status_code == 400 or status_code == 422:
            raise ValidationError(f"Validation error: {error_message}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {error_message}")
        elif status_code >= 500:
            raise ServerError(
                f"Server error ({status_code}): {error_message}",
                status_code=status_code,
                response_body=body,
            )
        else:
            raise APIError(
                f"API error ({status_code}): {error_message}",
                status_code=status_code,
                response_body=body,
            )

    def request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        use_org_base: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/jobs" or "jobs")
            json: JSON payload for the request
            params: Query parameters
            files: Files to upload
            use_org_base: Whether to use the organization-scoped base URL
            timeout: Override default timeout

        Returns:
            Response JSON as dictionary

        Raises:
            Various NscaleError subclasses on failure
        """
        # Clean up endpoint
        endpoint = endpoint.lstrip("/")

        # Build full URL
        if use_org_base:
            url = f"{self.org_base_url}/{endpoint}"
        else:
            url = f"{self.base_url}/{endpoint}"

        # Prepare request kwargs
        request_kwargs = {
            "timeout": timeout or self.timeout,
        }

        if json is not None:
            request_kwargs["json"] = json
        if params is not None:
            request_kwargs["params"] = params
        # Make request
        if files is not None:
            # File uploads: don't use session (to avoid Content-Type header issues)
            # Make a direct request with just Authorization header
            request_kwargs["files"] = files
            request_kwargs["headers"] = {
                "Authorization": self.session.headers["Authorization"]
            }
            response = requests.request(method, url, **request_kwargs)
        else:
            # Regular requests: use session
            response = self.session.request(method, url, **request_kwargs)

        # Handle errors
        if not response.ok:
            self._handle_error_response(response)

        # Parse and return response
        try:
            return response.json()
        except Exception:
            # If response is not JSON, return empty dict
            return {}

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make a GET request."""
        return self.request("GET", endpoint, params=params, **kwargs)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make a POST request."""
        return self.request("POST", endpoint, json=json, **kwargs)

    def put(self, endpoint: str, json: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make a PUT request."""
        return self.request("PUT", endpoint, json=json, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
