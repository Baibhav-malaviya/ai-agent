import json
import os
from datetime import datetime
from langchain_core.tools import tool


@tool
def sign_up(name: str, email: str) -> str:
    """
    Register a new user with their name and email address.

    Args:
        name: Full name of the user (first and last name)
        email: Valid email address of the user

    Returns:
        Success message with user details or error message if registration fails
    """
    # Input validation
    if not name or not name.strip():
        return "Error: Name cannot be empty. Please provide a valid name."

    if not email or not email.strip():
        return "Error: Email cannot be empty. Please provide a valid email."

    # Basic email validation
    if "@" not in email or "." not in email.split("@")[-1]:
        return f"Error: '{email}' is not a valid email address. Please provide a valid email."

    # Prepare user data
    new_user = {
        "name": name.strip(),
        "email": email.strip().lower(),
        "registered_at": datetime.now().isoformat(),
    }

    # Check if file exists and read existing users to prevent duplicates
    file_path = "registered_users.jsonl"
    existing_users = []

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        existing_users.append(json.loads(line))
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted, continue anyway
            print(f"Warning: Could not read existing users: {e}")

    # Check for duplicate email
    for user in existing_users:
        if user.get("email", "").lower() == new_user["email"]:
            return f"Error: A user with email '{email}' is already registered."

    # Write new user to file
    try:
        with open(file_path, 'a') as file:
            file.write(json.dumps(new_user) + "\n")

        return f"✅ Success! User '{name}' has been registered with email '{email}'."

    except IOError as e:
        return f"Error: Failed to register user due to file error: {str(e)}"