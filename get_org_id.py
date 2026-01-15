#!/usr/bin/env python3
"""
Helper script to retrieve your Nscale organization ID from the identity API.
"""

import os
import sys
import requests
from dotenv import load_dotenv


def get_organizations_from_api(api_token: str) -> list[dict]:
    """
    Fetch organizations from the Nscale identity API.

    Args:
        api_token: Nscale API token

    Returns:
        List of organization dictionaries with 'id' field

    Raises:
        Exception: If API request fails
    """
    base_url = "https://identity.nks.europe-west4.nscale.com"
    url = f"{base_url}/api/v1/acl"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        organizations = data.get("organizations", [])

        return organizations
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch organizations from API: {e}")


def update_env_file(org_id: str) -> None:
    """
    Update the .env file with the organization ID.

    Args:
        org_id: Organization ID to set
    """
    env_content = ""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            env_content = f.read()

    # Check if ORGANIZATION_ID already exists
    if "ORGANIZATION_ID=" in env_content:
        # Update existing
        lines = env_content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("ORGANIZATION_ID="):
                lines[i] = f"ORGANIZATION_ID={org_id}"
        env_content = "\n".join(lines)
    else:
        # Append new
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_content += f"ORGANIZATION_ID={org_id}\n"

    with open(".env", "w") as f:
        f.write(env_content)


def main():
    # Load environment variables
    load_dotenv()

    print()
    print("=" * 60)
    print("Nscale Organization ID Setup")
    print("=" * 60)
    print()

    # Check if .env exists
    if not os.path.exists(".env"):
        print("⚠ Warning: .env file not found!")
        print("Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Nscale API Configuration\n")
            f.write("NSCALE_TOKEN=your_token_here\n")
            f.write("ORGANIZATION_ID=your_org_id_here\n")
        print("✓ .env file created. Please add your NSCALE_TOKEN.\n")
        print("Please update NSCALE_TOKEN in .env and run this script again.")
        return

    # Get API token from environment
    api_token = os.getenv("NSCALE_TOKEN")
    if not api_token or api_token == "your_token_here":
        print("✗ Error: NSCALE_TOKEN not set in .env file")
        print()
        print("Please set your NSCALE_TOKEN in .env:")
        print("  NSCALE_TOKEN=your_actual_token")
        print()
        sys.exit(1)

    # Fetch organizations from API
    print("Fetching organizations from identity API...")
    print(f"API: https://identity.nks.europe-west4.nscale.com/api/v1/acl")
    print()

    try:
        organizations = get_organizations_from_api(api_token)

        if not organizations:
            print("✗ No organizations found for this token.")
            print()
            print("Please verify your NSCALE_TOKEN is correct.")
            sys.exit(1)

        # List all organizations
        print(f"Found {len(organizations)} organization(s):")
        print()
        for i, org in enumerate(organizations, 1):
            org_id = org.get("id", "N/A")
            num_projects = len(org.get("projects", []))
            num_endpoints = len(org.get("endpoints", []))
            print(f"  {i}. {org_id}")
            print(f"     Projects: {num_projects}, Endpoints: {num_endpoints}")
        print()

        # Pick the first organization
        selected_org = organizations[0]
        org_id = selected_org["id"]

        print(f"Selecting organization: {org_id}")
        print()

        # Update .env file
        update_env_file(org_id)

        print("=" * 60)
        print("✓ .env file updated successfully!")
        print("=" * 60)
        print(f"  ORGANIZATION_ID={org_id}")
        print()
        print("You can now run: make data")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
