import os
import pytest
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, MultiTurnSample
from ragas.llms import LangchainLLMWrapper

from utilities.assertions import Assertions
from utilities.ironman import IronMan
from utilities.logger import LoggerFactory
from utilities.email_reporter import PytestEmailReporter

# ---------------------------------------
# Load .env variables
# ---------------------------------------
load_dotenv()

# ---------------------------------------
# Pytest options
# ---------------------------------------
def pytest_addoption(parser):
    parser.addini("llm_model", help="LLM model name", default="gpt-4o-mini")
    parser.addini("llm_temperature", help="LLM temperature", default="0")

# ---------------------------------------
# Logger fixture
# ---------------------------------------
@pytest.fixture(scope="session")
def logger():
    return LoggerFactory.get_logger("pytest_logger")

@pytest.fixture
def assertions(logger):
    return Assertions(logger)

# ---------------------------------------
# LLM Wrapper fixture
# ---------------------------------------
@pytest.fixture(scope="session")
def get_llm_wrapper(request):
    model_name = request.config.getini("llm_model")
    temperature = float(request.config.getini("llm_temperature"))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY missing in .env")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )
    return LangchainLLMWrapper(llm)

# ---------------------------------------
# RAGAS Single Turn Fixture
# ---------------------------------------
@pytest.fixture
def get_singleturn_data(request, logger):
    test_data = request.param
    logger.info("Preparing SingleTurnSample data...")

    ironman = IronMan(logger)
    response_dictionary = ironman.get_api_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["user_input"],
        retrieved_contexts=[response_dictionary["response"]],
        response="\n".join(test_data["reference_contexts"]),
        reference=test_data["reference"]
    )
    return sample

# ---------------------------------------
# RAGAS Multi Turn Fixture
# ---------------------------------------
@pytest.fixture
def get_multiturn_data(request, logger):
    test_data = request.param
    logger.info("Preparing MultiTurnSample data...")

    ironman = IronMan(logger)

    question_chathistory = ironman.get_multiturn_conversation_data(
        test_data, as_object=True
    )
    (
        conversation,
        reference,
        reference_contexts,
        synthesizer_name
    ) = ironman.get_multiturn_conversation_data(test_data, as_object=False)

    logger.info(question_chathistory)

    response_dict = ironman.get_rahul_shetty_llm_api_response(question_chathistory)

    user_inputs = [msg for msg in conversation]

    sample = MultiTurnSample(
        user_input=user_inputs,
        retrieved_contexts=[
            doc["page_content"]
            for doc in response_dict.get("retrieved_docs", [])
        ],
        response=[response_dict.get("answer", "")],
        references=test_data.get("reference", "")
    )

    response = [response_dict.get("answer", "")]
    return sample, response, question_chathistory

# ---------------------------------------
# Clean pytest metadata
# ---------------------------------------
def pytest_metadata(metadata):
    metadata.pop("Packages", None)
    metadata.pop("Plugins", None)

# =====================================================================
# EMAIL REPORTER HOOKS
# =====================================================================
def pytest_sessionstart(session):
    logger = LoggerFactory.get_logger("email-reporter")
    reporter = PytestEmailReporter(logger)

    session.email_reporter = reporter

    # Register plugin
    session.config.pluginmanager.register(reporter, name="email_reporter_plugin")

    reporter.session_start()


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session, "email_reporter"):
        reporter = session.email_reporter
        reporter.session_end()
        reporter.send_email()
        session.config.pluginmanager.unregister(reporter)
