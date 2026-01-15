"""
Custom exceptions for Nscale API client.
"""


class NscaleError(Exception):
    """Base exception for all Nscale API errors."""
    pass


class AuthenticationError(NscaleError):
    """Authentication failed (invalid token, missing credentials, etc.)."""
    pass


class ValidationError(NscaleError):
    """Request validation failed (invalid parameters, missing fields, etc.)."""
    pass


class ResourceNotFoundError(NscaleError):
    """Requested resource not found (404)."""
    pass


class RateLimitError(NscaleError):
    """API rate limit exceeded (429)."""
    pass


class ServerError(NscaleError):
    """Server-side error (500, 502, 503, etc.)."""

    def __init__(self, message: str, status_code: int = 500, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class APIError(NscaleError):
    """Generic API error."""

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
