"""
Base classes for Nscale services.
"""

from .platform import NscaleClient


class BaseService:
    """
    Base class for all Nscale service clients.

    Provides common initialization and access to the underlying
    NscaleClient for making API requests.
    """

    def __init__(self, client: NscaleClient):
        """
        Initialize service with Nscale client.

        Args:
            client: Initialized NscaleClient instance
        """
        self.client = client
