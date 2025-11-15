import json
import sys
from pathlib import Path
from ragas.messages import HumanMessage, AIMessage
import pytest
import requests
import allure


class IronMan:
    """
    Utility class providing reusable helper methods for:
      - Loading test data from JSON files
      - Sending API requests and logging responses
      - Dynamically determining the project root directory
      - Loading multi-turn conversation datasets
    """

    # -----------------------------------------------------------------------------
    # 🔧 INITIALIZER
    # -----------------------------------------------------------------------------
    def __init__(self, logger):
        self.logger = logger

    # -----------------------------------------------------------------------------
    # 📂 LOAD TEST DATA (Single-turn or Dataset JSON)
    # -----------------------------------------------------------------------------
    @staticmethod
    def load_test_data(json_path: str, data_set: str = None):
        """
        Loads test data from a JSON file located within the project directory.

        Args:
            json_path (str): Relative path to the JSON file from the project root.
            data_set (str, optional): Name of the dataset folder. Defaults to None.

        Returns:
            dict: Parsed JSON data as a Python dictionary.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        # Construct the full file path
        if data_set:
            file_path = IronMan.get_sys_root() / "dataset" / json_path / f"{data_set}_dataset.json"
        else:
            file_path = IronMan.get_sys_root() / "dataset" / json_path / f"{json_path}_dataset.json"

        # Read and parse the JSON file
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------------
    # 🌐 SEND API REQUEST AND CAPTURE RESPONSE
    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    # 🌐 SEND API REQUEST AND CAPTURE RESPONSE
    # -----------------------------------------------------------------------------
    def get_rahul_shetty_llm_api_response(self, test_data: dict):
        """
        Sends the user input from the test data to a remote API endpoint
        and logs the received response.

        Args:
            test_data (dict): A dictionary containing the 'question' key.

        Returns:
            dict: Parsed JSON response from the API.
        """
        # Send POST request to API and get JSON response
        payload = {
            "question": test_data.get("question"),
            "chat_history": test_data.get("chat_history", [])
        }
        response_dictionary = requests.post(
            url="https://rahulshettyacademy.com/rag-llm/ask",
            json=payload
        ).json()

        # Log the API interaction
        self.logger.info("---------------- API RESPONSE -------------------")
        self.logger.info(response_dictionary)

        with allure.step("Send user input to API and capture response"):
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

    # -----------------------------------------------------------------------------
    # 📁 DETERMINE PROJECT ROOT DIRECTORY
    # -----------------------------------------------------------------------------
    @staticmethod
    def get_sys_root() -> Path:
        """
        Determines the root directory of the project dynamically using sys.path.

        Returns:
            Path: The resolved absolute path of the project's root directory.
        """
        return Path(sys.path[0]).resolve()

    # ===============================================================
    # 🧩 Multi-turn dataset loader
    # ===============================================================
    def get_multiturn_conversation_data(self, test_data: dict, as_object: bool = False):
        """
        Processes a multi-turn conversation JSON object and converts it into
        LangChain-compatible message objects or question/chat_history format.

        Args:
            test_data (dict): JSON object containing conversation data.
            as_object (bool): If True, returns a dict with "question" and "chat_history" keys.

        Returns:
            tuple or dict:
                - Default: (conversation, reference, reference_contexts, synthesizer_name)
                - If as_object=True:
                    {
                        "question": str,
                        "chat_history": list[dict]
                    }
        """

        # ---------------------------------------------------------------
        # Validate structure
        # ---------------------------------------------------------------
        if not isinstance(test_data, dict):
            raise TypeError("❌ test_data must be a dictionary object.")
        if "conversation" not in test_data:
            raise ValueError("❌ test_data must contain a 'conversation' key.")
        if not test_data["conversation"]:
            raise ValueError("❌ 'conversation' array is empty.")

        # ---------------------------------------------------------------
        # Build conversation array of HumanMessage / AIMessage
        # ---------------------------------------------------------------
        conversation = []
        for msg in test_data["conversation"]:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "human":
                conversation.append(HumanMessage(content=content))
            elif role == "ai":
                conversation.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role '{role}' found in conversation.")

        # ---------------------------------------------------------------
        # Extract reference details
        # ---------------------------------------------------------------
        reference = test_data.get("reference", "")
        reference_contexts = test_data.get("reference_contexts", [])
        synthesizer_name = test_data.get("synthesizer_name", "")

        # ---------------------------------------------------------------
        # If as_object=True → build question + chat_history format
        # ---------------------------------------------------------------
        if as_object:
            conversation_items = test_data["conversation"]

            # First human = question
            first_human = next(
                (item for item in conversation_items if item.get("role") == "human"),
                None
            )
            if not first_human:
                raise ValueError("No human message found in conversation.")

            question = first_human.get("content", "")
            first_human_index = conversation_items.index(first_human)
            chat_history = conversation_items[first_human_index + 1:]

            return {
                "question": question,
                "chat_history": chat_history
            }

        # ---------------------------------------------------------------
        # Default return (LangChain messages + metadata)
        # ---------------------------------------------------------------
        return conversation, reference, reference_contexts, synthesizer_name
