import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, MultiTurnSample
from ragas.llms import LangchainLLMWrapper

from utilities.ironman import IronMan
from utilities.logger import LoggerFactory


# -------------------------------
# Load environment variables from .env
# -------------------------------
load_dotenv()  # ✅ Automatically loads OPENAI_API_KEY and others from your .env file

# -------------------------------
# Add custom pytest options
# -------------------------------
def pytest_addoption(parser):
    parser.addini("llm_model", help="LLM model name to use", default="gpt-4o-mini")
    parser.addini("llm_temperature", help="LLM temperature setting", default="0")


# -------------------------------
# Configure allure report directory
# -------------------------------
def pytest_configure(config):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    allure_dir = os.path.join(root_dir, "testreports", "allure-results")
    os.makedirs(allure_dir, exist_ok=True)
    config.option.allure_report_dir = allure_dir


# -------------------------------
# Logger fixture
# -------------------------------
@pytest.fixture(scope="session")
def logger():
    return LoggerFactory.get_logger("pytest_logger")


# -------------------------------
# LLM Wrapper fixture (reads model + temperature from pytest.ini)
# -------------------------------
@pytest.fixture(scope="session")
def get_llm_wrapper(request):
    """
    Creates a LangChain LLM wrapper for use in RAGAS metrics evaluation.
    Model name and temperature are read from pytest.ini.
    OpenAI API key is loaded from .env file.
    """
    model_name = request.config.getini("llm_model")
    temperature = float(request.config.getini("llm_temperature"))

    # ✅ Ensure API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

    # ✅ Initialize ChatOpenAI with API key
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm

# -------------------------------
# get single turn data fixture
# -------------------------------
@pytest.fixture
def get_singleturn_data(request, logger):
    """
    Fixture to prepare SingleTurnSample data for RAGAS evaluation.
    Automatically available to all test files.
    """
    test_data = request.param
    logger.info("Preparing data sample for singleturn RAGAS evaluation...")

    ironman = IronMan(logger)
    response_dictionary = ironman.get_api_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["user_input"],
        retrieved_contexts=[response_dictionary["response"]],
        response="\n".join(test_data["reference_contexts"]),
        reference=test_data["reference"]
    )
    return sample

# -------------------------------
# get multi turn data fixture
# -------------------------------
@pytest.fixture
def get_multiturn_data(request, logger):
    """
    Fixture to prepare MultiTurnSample data for RAGAS evaluation.
    Automatically available to all test files.
    """
    test_data = request.param
    logger.info("Preparing data sample for MultiTurn RAGAS evaluation...")

    ironman = IronMan(logger)
    response_dictionary = ironman.get_api_response(test_data)

    # Create multi-turn sample
    sample = MultiTurnSample(
        user_inputs=test_data["user_inputs"],               # list of queries
        retrieved_contexts=response_dictionary["responses"], # list of contexts
        response="\n".join(test_data["reference_contexts"]), # list of model responses
        references=test_data["references"]                  # list of reference answers
    )
    return sample

