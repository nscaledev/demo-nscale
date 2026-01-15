"""
Test suite for get_org_id.py module.

BDD Scenarios:
==============

Feature: Automatic Organization ID Discovery
  As a developer setting up Nscale
  I want to automatically fetch my organization ID from the API
  So that I don't need to manually look it up in the console

  Scenario: Fetch organizations successfully
    Given a valid API token
    When I call get_organizations_from_api()
    Then it should return a list of organizations

  Scenario: Handle API authentication error
    Given an invalid API token
    When I call get_organizations_from_api()
    Then an exception should be raised with authentication error

  Scenario: Handle API network error
    Given a network failure
    When I call get_organizations_from_api()
    Then an exception should be raised with connection error

  Scenario: Parse organization response correctly
    Given an ACL response with multiple organizations
    When I parse the response
    Then it should extract organization IDs, projects, and endpoints

  Scenario: Update .env file with organization ID
    Given an organization ID
    When I call update_env_file()
    Then the .env file should be updated with ORGANIZATION_ID

  Scenario: Update existing ORGANIZATION_ID in .env
    Given .env file with existing ORGANIZATION_ID
    When I call update_env_file() with new ID
    Then the existing ID should be replaced

  Scenario: Append ORGANIZATION_ID to .env without it
    Given .env file without ORGANIZATION_ID
    When I call update_env_file()
    Then ORGANIZATION_ID should be appended
"""

import pytest
import tempfile
import os

# Import functions from get_org_id.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from get_org_id import get_organizations_from_api, update_env_file


class TestGetOrganizationsFromAPI:
    """Test fetching organizations from identity API."""

    def test_fetch_organizations_successfully(self, requests_mock):
        """
        Scenario: Fetch organizations successfully
        """
        # Arrange
        api_token = "test_token_123"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        mock_response = {
            "organizations": [
                {
                    "id": "org_123",
                    "endpoints": [{"name": "endpoint1", "operations": ["read"]}],
                    "projects": [{"id": "proj_1", "endpoints": []}]
                },
                {
                    "id": "org_456",
                    "endpoints": [],
                    "projects": []
                }
            ]
        }

        requests_mock.get(url, json=mock_response)

        # Act
        organizations = get_organizations_from_api(api_token)

        # Assert
        assert requests_mock.called
        assert requests_mock.last_request.headers["Authorization"] == "Bearer test_token_123"
        assert len(organizations) == 2
        assert organizations[0]["id"] == "org_123"
        assert organizations[1]["id"] == "org_456"
        assert len(organizations[0]["projects"]) == 1
        assert len(organizations[0]["endpoints"]) == 1

    def test_handle_authentication_error(self, requests_mock):
        """
        Scenario: Handle API authentication error
        """
        # Arrange
        api_token = "invalid_token"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        requests_mock.get(url, status_code=401, text="Unauthorized")

        # Act & Assert
        with pytest.raises(Exception, match="Failed to fetch organizations"):
            get_organizations_from_api(api_token)

    def test_handle_network_error(self, requests_mock):
        """
        Scenario: Handle API network error
        """
        # Arrange
        import requests
        api_token = "test_token"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        requests_mock.get(url, exc=requests.exceptions.ConnectionError("Network error"))

        # Act & Assert
        with pytest.raises(Exception, match="Failed to fetch organizations"):
            get_organizations_from_api(api_token)

    def test_handle_server_error(self, requests_mock):
        """
        Scenario: Handle API server error
        """
        # Arrange
        api_token = "test_token"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        requests_mock.get(url, status_code=500, text="Internal Server Error")

        # Act & Assert
        with pytest.raises(Exception, match="Failed to fetch organizations"):
            get_organizations_from_api(api_token)

    def test_parse_empty_organizations(self, requests_mock):
        """
        Scenario: Handle response with no organizations
        """
        # Arrange
        api_token = "test_token"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        requests_mock.get(url, json={"organizations": []})

        # Act
        organizations = get_organizations_from_api(api_token)

        # Assert
        assert organizations == []

    def test_parse_missing_organizations_key(self, requests_mock):
        """
        Scenario: Handle response without organizations key
        """
        # Arrange
        api_token = "test_token"
        url = "https://identity.nks.europe-west4.nscale.com/api/v1/acl"

        requests_mock.get(url, json={"data": "something_else"})

        # Act
        organizations = get_organizations_from_api(api_token)

        # Assert
        assert organizations == []


class TestUpdateEnvFile:
    """Test updating .env file with organization ID."""

    def test_update_existing_organization_id(self):
        """
        Scenario: Update existing ORGANIZATION_ID in .env
        """
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write("NSCALE_TOKEN=token_123\n")
            f.write("ORGANIZATION_ID=old_org_id\n")
            f.write("OTHER_VAR=value\n")
            env_file = f.name

        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(env_file)
        os.chdir(temp_dir)
        os.rename(env_file, ".env")

        try:
            # Act
            update_env_file("new_org_id")

            # Assert
            with open(".env", "r") as f:
                content = f.read()

            assert "ORGANIZATION_ID=new_org_id" in content
            assert "ORGANIZATION_ID=old_org_id" not in content
            assert "NSCALE_TOKEN=token_123" in content
            assert "OTHER_VAR=value" in content
        finally:
            # Cleanup
            os.chdir(original_cwd)
            if os.path.exists(os.path.join(temp_dir, ".env")):
                os.remove(os.path.join(temp_dir, ".env"))

    def test_append_organization_id_to_env(self):
        """
        Scenario: Append ORGANIZATION_ID to .env without it
        """
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write("NSCALE_TOKEN=token_123\n")
            env_file = f.name

        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(env_file)
        os.chdir(temp_dir)
        os.rename(env_file, ".env")

        try:
            # Act
            update_env_file("new_org_id")

            # Assert
            with open(".env", "r") as f:
                content = f.read()

            assert "ORGANIZATION_ID=new_org_id" in content
            assert "NSCALE_TOKEN=token_123" in content
            assert content.count("ORGANIZATION_ID") == 1
        finally:
            # Cleanup
            os.chdir(original_cwd)
            if os.path.exists(os.path.join(temp_dir, ".env")):
                os.remove(os.path.join(temp_dir, ".env"))

    def test_create_env_if_not_exists(self):
        """
        Scenario: Create .env file if it doesn't exist
        """
        # Arrange
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Act
            update_env_file("test_org_id")

            # Assert
            assert os.path.exists(".env")
            with open(".env", "r") as f:
                content = f.read()

            assert "ORGANIZATION_ID=test_org_id" in content
        finally:
            # Cleanup
            os.chdir(original_cwd)
            if os.path.exists(os.path.join(temp_dir, ".env")):
                os.remove(os.path.join(temp_dir, ".env"))
            os.rmdir(temp_dir)

    def test_preserve_env_file_formatting(self):
        """
        Scenario: Preserve other variables when updating
        """
        # Arrange
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            f.write("# Configuration\n")
            f.write("NSCALE_TOKEN=token\n")
            f.write("\n")
            f.write("# Old org\n")
            f.write("ORGANIZATION_ID=old\n")
            f.write("DEBUG=true\n")
            env_file = f.name

        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(env_file)
        os.chdir(temp_dir)
        os.rename(env_file, ".env")

        try:
            # Act
            update_env_file("new_org")

            # Assert
            with open(".env", "r") as f:
                content = f.read()

            assert "# Configuration" in content
            assert "NSCALE_TOKEN=token" in content
            assert "ORGANIZATION_ID=new_org" in content
            assert "DEBUG=true" in content
            assert "ORGANIZATION_ID=old" not in content
        finally:
            # Cleanup
            os.chdir(original_cwd)
            if os.path.exists(os.path.join(temp_dir, ".env")):
                os.remove(os.path.join(temp_dir, ".env"))
