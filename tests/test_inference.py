"""
Test suite for nscale.inference module.

BDD Scenarios:
==============

Feature: Inference with Deployed Models
  As a developer running predictions
  I want to use deployed models for inference
  So that I can generate responses to queries

  Scenario: Initialize inference client
    Given a valid API token
    When I create an InferenceClient
    Then it should be configured with correct headers and base URL

  Scenario: Initialize without API token
    Given no API token
    When I create an InferenceClient
    Then an NscaleError should be raised

  Scenario: List available inference models
    Given models are deployed
    When I call list_models()
    Then it should return the list of available models

  Scenario: Generate chat completion successfully
    Given a valid model ID and messages
    When I call chat_completion()
    Then it should send the request with correct parameters
    And return the completion response

  Scenario: Generate chat completion with custom parameters
    Given custom temperature and max_tokens
    When I call chat_completion()
    Then it should use the custom parameters

  Scenario: Generate answer convenience method
    Given a model and question
    When I call generate_answer()
    Then it should format the request correctly
    And extract the answer from the response

  Scenario: Generate answer with system prompt
    Given a system prompt is provided
    When I call generate_answer()
    Then the system prompt should be included in messages

  Scenario: Handle inference API errors
    Given an invalid request
    When the API returns an error
    Then an NscaleError should be raised with details

  Scenario: Use client as context manager
    Given an InferenceClient instance
    When I use it in a with statement
    Then it should properly close the session on exit
"""

import pytest
from unittest.mock import Mock, patch

from nscale.inference import InferenceClient
from nscale.exceptions import NscaleError


class TestInferenceClientInitialization:
    """Test inference client initialization."""

    def test_initialize_with_valid_token(self):
        """
        Scenario: Initialize inference client
        """
        # Arrange
        api_token = "test_token_abc"
        base_url = "https://inference.api.example.com/v1"

        # Act
        client = InferenceClient(api_token=api_token, base_url=base_url)

        # Assert
        assert client.api_token == api_token
        assert client.base_url == "https://inference.api.example.com/v1"
        assert client.session.headers["Authorization"] == f"Bearer {api_token}"
        assert client.session.headers["Content-Type"] == "application/json"

    def test_initialize_without_token_raises_error(self):
        """
        Scenario: Initialize without API token
        """
        # Arrange
        api_token = None

        # Act & Assert
        with pytest.raises(NscaleError, match="API token is required"):
            InferenceClient(api_token=api_token)


class TestInferenceClientListModels:
    """Test listing available models."""

    def test_list_models_successfully(self, requests_mock):
        """
        Scenario: List available inference models
        """
        # Arrange
        client = InferenceClient(api_token="token_123")

        url = "https://inference.api.nscale.com/v1/models"
        requests_mock.get(url, json={
            "data": [
                {"id": "model_1", "object": "model", "owned_by": "OpenAI"},
                {"id": "model_2", "object": "model", "owned_by": "Qwen"}
            ]
        })

        # Act
        result = client.list_models()

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.method == "GET"
        assert requests_mock.last_request.url == url
        assert requests_mock.last_request.headers["Authorization"] == "Bearer token_123"
        assert len(result) == 2
        assert result[0]["id"] == "model_1"

    def test_list_models_handles_error(self, requests_mock):
        """
        Scenario: Handle list models API error
        """
        # Arrange
        client = InferenceClient(api_token="token_123")

        url = "https://inference.api.nscale.com/v1/models"
        requests_mock.get(url, status_code=500, text="Server error")

        # Act & Assert
        with pytest.raises(NscaleError, match="Failed to list models"):
            client.list_models()


class TestInferenceClientChatCompletion:
    """Test chat completion functionality."""

    def test_chat_completion_successfully(self, requests_mock):
        """
        Scenario: Generate chat completion successfully
        """
        # Arrange
        client = InferenceClient(api_token="token_123")

        url = "https://inference.api.nscale.com/v1/chat/completions"
        requests_mock.post(url, json={
            "id": "chatcmpl_123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is the answer"
                    }
                }
            ]
        })

        messages = [{"role": "user", "content": "What is AI?"}]

        # Act
        result = client.chat_completion(
            model="test-model",
            messages=messages
        )

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.method == "POST"
        assert requests_mock.last_request.url == url
        assert requests_mock.last_request.json()["model"] == "test-model"
        assert requests_mock.last_request.json()["messages"] == messages
        assert requests_mock.last_request.headers["Authorization"] == "Bearer token_123"
        assert result["choices"][0]["message"]["content"] == "This is the answer"

    def test_chat_completion_with_custom_parameters(self, requests_mock):
        """
        Scenario: Generate chat completion with custom parameters
        """
        # Arrange
        client = InferenceClient(api_token="token_123")

        url = "https://inference.api.nscale.com/v1/chat/completions"
        requests_mock.post(url, json={"choices": [{"message": {"content": "response"}}]})

        # Act
        client.chat_completion(
            model="gpt-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2048,
            temperature=0.8,
            top_p=0.9
        )

        # Assert
        assert requests_mock.called
        payload = requests_mock.last_request.json()
        assert payload["max_tokens"] == 2048
        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.9

    def test_chat_completion_handles_error(self, requests_mock):
        """
        Scenario: Handle inference API errors
        """
        # Arrange
        client = InferenceClient(api_token="token_123")

        url = "https://inference.api.nscale.com/v1/chat/completions"
        requests_mock.post(url, status_code=400, text="Bad request")

        # Act & Assert
        with pytest.raises(NscaleError, match="Chat completion failed"):
            client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "test"}]
            )


class TestInferenceClientGenerateAnswer:
    """Test generate_answer convenience method."""

    @patch.object(InferenceClient, 'chat_completion')
    def test_generate_answer_successfully(self, mock_chat_completion):
        """
        Scenario: Generate answer convenience method
        """
        # Arrange
        mock_chat_completion.return_value = {
            "choices": [
                {"message": {"content": "Machine learning is a subset of AI"}}
            ]
        }

        client = InferenceClient(api_token="token_123")

        # Act
        result = client.generate_answer(
            model="qwen-model",
            question="What is machine learning?"
        )

        # Assert
        mock_chat_completion.assert_called_once()
        call_args = mock_chat_completion.call_args

        assert call_args[1]["model"] == "qwen-model"
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is machine learning?"
        assert result == "Machine learning is a subset of AI"

    @patch.object(InferenceClient, 'chat_completion')
    def test_generate_answer_with_system_prompt(self, mock_chat_completion):
        """
        Scenario: Generate answer with system prompt
        """
        # Arrange
        mock_chat_completion.return_value = {
            "choices": [{"message": {"content": "Detailed explanation"}}]
        }

        client = InferenceClient(api_token="token_123")

        # Act
        result = client.generate_answer(
            model="model-id",
            question="Explain this concept",
            system_prompt="You are a helpful teacher"
        )

        # Assert
        call_args = mock_chat_completion.call_args
        messages = call_args[1]["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful teacher"
        assert messages[1]["role"] == "user"

    @patch.object(InferenceClient, 'chat_completion')
    def test_generate_answer_raises_error_on_no_choices(self, mock_chat_completion):
        """
        Scenario: Handle missing choices in response
        """
        # Arrange
        mock_chat_completion.return_value = {"choices": []}

        client = InferenceClient(api_token="token_123")

        # Act & Assert
        with pytest.raises(NscaleError, match="No completion choices"):
            client.generate_answer(model="model", question="test")


class TestInferenceClientContextManager:
    """Test context manager functionality."""

    def test_use_client_as_context_manager(self):
        """
        Scenario: Use client as context manager
        """
        # Arrange
        client = InferenceClient(api_token="token_123")
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
        client = InferenceClient(api_token="token_123")
        client.session.close = Mock()

        # Act & Assert
        try:
            with client:
                raise ValueError("Test error")
        except ValueError:
            pass

        client.session.close.assert_called_once()
