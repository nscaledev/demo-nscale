#!/usr/bin/env python3
"""
Helper script to retrieve your Nscale organization ID.
"""

import os
import re
from dotenv import load_dotenv


def get_organization_id_from_console() -> str:
    """
    Nscale doesn't provide an API endpoint to list organizations.
    Users must retrieve their organization ID from the console.

    Returns the organization ID entered by the user.
    """
    print("=" * 60)
    print("How to find your Nscale Organization ID:")
    print("=" * 60)
    print()
    print("1. Visit: https://console.nscale.com/settings")
    print("2. Look for 'Organization ID' in your account settings")
    print("3. Copy the UUID (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    print()
    print("=" * 60)
    print()

    org_id = input("Enter your Organization ID: ").strip()
    return org_id


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
        print("✓ .env file created. Please add your credentials.\n")

    # Get organization ID from user
    org_id = get_organization_id_from_console()

    if not org_id:
        print("⚠ No organization ID entered. Exiting.")
        return

    # Validate format (should be a UUID)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, org_id, re.IGNORECASE):
        print(f"⚠ Warning: '{org_id}' doesn't look like a valid UUID.")
        print("   Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return

    # Update .env file
    try:
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

        print()
        print("=" * 60)
        print("✓ .env file updated successfully!")
        print("=" * 60)
        print(f"  ORGANIZATION_ID={org_id}")
        print()
        print("You can now run: make finetune")
        print()
    except Exception as e:
        print(f"✗ Error updating .env file: {e}")


if __name__ == "__main__":
    main()
