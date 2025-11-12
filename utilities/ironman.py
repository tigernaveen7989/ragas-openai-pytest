import json
import sys
from pathlib import Path

import pytest
import requests
import allure


class IronMan:
    """
    Utility class providing reusable helper methods for:
      - Loading test data from JSON files
      - Sending API requests and logging responses
      - Dynamically determining the project root directory
    """
    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def load_test_data(json_path: str):
        """
        Loads test data from a JSON file located within the project directory.

        Args:
            json_path (str): Relative path to the JSON file from the project root.

        Returns:
            dict: Parsed JSON data as a Python dictionary.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        # Construct the full file path relative to the project root directory
        file_path = IronMan.get_sys_root() / "dataset" / json_path / f"{json_path}_dataset.json"

        # Read and parse the JSON file
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def get_api_response(self, test_data: dict):
        """
        Sends the user input from the test data to a remote API endpoint
        and logs the received response.

        Args:
            test_data (dict): A dictionary containing the 'user_input' key.
            logger: Logger instance used to log API requests and responses.

        Returns:
            dict: Parsed JSON response from the API.
        """
        # Send POST request to API and get JSON response
        response_dictionary = requests.post(
            url="https://apifreellm.com/api/chat",
            json={"message": test_data["user_input"]}
        ).json()

        # Log the API interaction
        self.logger.info("---------------- API RESPONSE -------------------")
        self.logger.info(response_dictionary)

        with allure.step("Send user input to API and capture response"):
            # Check for API errors
            if response_dictionary.get("status") == "error":
                allure.attach(
                    json.dumps(response_dictionary, indent=2),
                    name="API Error Response",
                    attachment_type=allure.attachment_type.JSON
                )
                pytest.skip("❌ Skipping test due to API internal server error.")
            else:
                response_text = response_dictionary.get("response", "⚠️ No 'response' key found in dictionary")
                allure.attach(
                    str(response_text),
                    name="API Response",
                    attachment_type=allure.attachment_type.TEXT
                )
        return response_dictionary

    @staticmethod
    def get_sys_root() -> Path:
        """
        Determines the root directory of the project dynamically using sys.path.

        Returns:
            Path: The resolved absolute path of the project's root directory.
        """
        # Typically, the first entry in sys.path corresponds to the project root
        return Path(sys.path[0]).resolve()
