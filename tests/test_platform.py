"""
Test suite for nscale.platform module.

BDD Scenarios:
==============

Feature: Platform Client Authentication and Error Handling
  As a developer using the Nscale API
  I want a robust client that handles authentication and errors gracefully
  So that I can build reliable applications

  Scenario: Initialize client with valid credentials
    Given valid API token and organization ID
    When I create a NscaleClient instance
    Then the client should be configured with correct headers and base URL

  Scenario: Handle authentication errors
    Given an invalid API token
    When I make an API request that returns 401
    Then an AuthenticationError should be raised with clear message

  Scenario: Handle validation errors
    Given invalid request parameters
    When I make an API request that returns 400
    Then a ValidationError should be raised with details

  Scenario: Handle resource not found errors
    Given a non-existent resource ID
    When I make an API request that returns 404
    Then a ResourceNotFoundError should be raised

  Scenario: Handle rate limit errors
    Given multiple rapid API requests
    When the API returns 429
    Then a RateLimitError should be raised

  Scenario: Handle server errors
    Given a valid request
    When the API returns 500
    Then a ServerError should be raised with status code

  Scenario: Retry on transient failures
    Given a request that initially fails with 503
    When the retry logic kicks in
    Then the request should be retried automatically

  Scenario: Make successful GET request
    Given a valid endpoint
    When I make a GET request
    Then it should return the JSON response

  Scenario: Make successful POST request with JSON payload
    Given a valid endpoint and JSON data
    When I make a POST request
    Then it should send the data and return the response

  Scenario: Use client as context manager
    Given a NscaleClient instance
    When I use it in a with statement
    Then it should properly close the session on exit
"""

import pytest
from unittest.mock import Mock

from nscale.platform import NscaleClient
from nscale.exceptions import (
    AuthenticationError,
    ValidationError,
    ResourceNotFoundError,
    RateLimitError,
    ServerError,
    APIError,
)


class TestNscaleClientInitialization:
    """Test NscaleClient initialization and configuration."""

    def test_initialize_with_valid_credentials(self):
        """
        Scenario: Initialize client with valid credentials
        """
        # Arrange
        api_token = "test_token_123"
        org_id = "org_abc_456"
        base_url = "https://api.example.com/v1"

        # Act
        client = NscaleClient(
            api_token=api_token,
            organization_id=org_id,
            base_url=base_url
        )

        # Assert
        assert client.api_token == api_token
        assert client.organization_id == org_id
        assert client.base_url == "https://api.example.com/v1"
        assert client.org_base_url == f"https://api.example.com/v1/organizations/{org_id}"
        assert client.session.headers["Authorization"] == f"Bearer {api_token}"
        assert client.session.headers["Content-Type"] == "application/json"

    def test_initialize_without_api_token_raises_error(self):
        """
        Scenario: Initialize without API token
        """
        # Arrange
        api_token = None
        org_id = "org_123"

        # Act & Assert
        with pytest.raises(AuthenticationError, match="API token is required"):
            NscaleClient(api_token=api_token, organization_id=org_id, base_url="https://api.example.com")

    def test_initialize_without_org_id_raises_error(self):
        """
        Scenario: Initialize without organization ID
        """
        # Arrange
        api_token = "token_123"
        org_id = None

        # Act & Assert
        with pytest.raises(ValidationError, match="Organization ID is required"):
            NscaleClient(api_token=api_token, organization_id=org_id, base_url="https://api.example.com")


class TestNscaleClientErrorHandling:
    """Test error handling for various HTTP status codes."""

    def test_handle_401_authentication_error(self):
        """
        Scenario: Handle authentication errors (401)
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.ok = False
        mock_response.text = '{"error": "Invalid token"}'
        mock_response.json.return_value = {"code": "unauthorized", "message": "Invalid token"}

        # Act & Assert
        with pytest.raises(AuthenticationError, match="Invalid token"):
            client._handle_error_response(mock_response)

    def test_handle_400_validation_error(self):
        """
        Scenario: Handle validation errors (400)
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.ok = False
        mock_response.text = '{"error": "Missing required field"}'
        mock_response.json.return_value = {"code": "validation_error", "message": "Missing required field"}

        # Act & Assert
        with pytest.raises(ValidationError, match="Missing required field"):
            client._handle_error_response(mock_response)

    def test_handle_404_resource_not_found(self):
        """
        Scenario: Handle resource not found (404)
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.ok = False
        mock_response.text = '{"error": "Resource not found"}'
        mock_response.json.return_value = {"code": "not_found", "message": "Resource not found"}

        # Act & Assert
        with pytest.raises(ResourceNotFoundError, match="Resource not found"):
            client._handle_error_response(mock_response)

    def test_handle_429_rate_limit_error(self):
        """
        Scenario: Handle rate limit errors (429)
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.ok = False
        mock_response.text = '{"error": "Rate limit exceeded"}'
        mock_response.json.return_value = {"code": "rate_limit", "message": "Rate limit exceeded"}

        # Act & Assert
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            client._handle_error_response(mock_response)

    def test_handle_500_server_error(self):
        """
        Scenario: Handle server errors (500)
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.text = '{"error": "Internal server error"}'
        mock_response.json.return_value = {"code": "internal", "message": "Internal server error"}

        # Act & Assert
        with pytest.raises(ServerError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 500
        assert "Internal server error" in str(exc_info.value)


class TestNscaleClientRequests:
    """Test making HTTP requests with the client."""

    def test_make_successful_get_request(self, requests_mock):
        """
        Scenario: Make successful GET request
        """
        # Arrange
        client = NscaleClient("token_123", "org_456", "https://api.example.com")

        url = "https://api.example.com/organizations/org_456/test-endpoint"
        requests_mock.get(url, json={"data": "test_data"})

        # Act
        result = client.get("test-endpoint", params={"limit": 10})

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.method == "GET"
        assert requests_mock.last_request.url == f"{url}?limit=10"
        assert requests_mock.last_request.headers["Authorization"] == "Bearer token_123"
        assert result == {"data": "test_data"}

    def test_make_successful_post_request(self, requests_mock):
        """
        Scenario: Make successful POST request with JSON
        """
        # Arrange
        client = NscaleClient("token_123", "org_456", "https://api.example.com")

        url = "https://api.example.com/organizations/org_456/create-endpoint"
        requests_mock.post(url, json={"id": "created_123"})
        payload = {"name": "test", "value": 42}

        # Act
        result = client.post("create-endpoint", json=payload)

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.method == "POST"
        assert requests_mock.last_request.json() == payload
        assert requests_mock.last_request.headers["Authorization"] == "Bearer token_123"
        assert result == {"id": "created_123"}

    def test_make_file_upload_request(self, requests_mock):
        """
        Scenario: Make file upload request (should not use session)
        """
        # Arrange
        client = NscaleClient("token_123", "org_456", "https://api.example.com")

        url = "https://api.example.com/organizations/org_456/files"
        requests_mock.post(url, json={"file_id": "file_789"})
        files = {"file": ("test.csv", b"content", "text/csv")}

        # Act
        result = client.request("POST", "files", files=files)

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.method == "POST"
        assert requests_mock.last_request.headers["Authorization"] == "Bearer token_123"
        # Content-Type should not include application/json for file uploads
        assert "application/json" not in requests_mock.last_request.headers.get("Content-Type", "")
        assert result == {"file_id": "file_789"}


class TestNscaleClientContextManager:
    """Test context manager functionality."""

    def test_use_client_as_context_manager(self):
        """
        Scenario: Use client as context manager
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        client.session.close = Mock()

        # Act
        with client as ctx:
            assert ctx == client

        # Assert
        client.session.close.assert_called_once()

    def test_context_manager_closes_on_exception(self):
        """
        Scenario: Context manager closes even on exception
        """
        # Arrange
        client = NscaleClient("token", "org_id", "https://api.example.com")
        client.session.close = Mock()

        # Act & Assert
        try:
            with client:
                raise ValueError("Test exception")
        except ValueError:
            pass

        client.session.close.assert_called_once()
