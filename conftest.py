import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from utilities.assertions import Assertions
from utilities.ironman import IronMan
from utilities.logger import LoggerFactory
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


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
    # ----- Existing Allure Setup (keep this unchanged) -----
    root_dir = os.path.dirname(os.path.abspath(__file__))
    allure_dir = os.path.join(root_dir, "testreports", "allure-results")
    os.makedirs(allure_dir, exist_ok=True)
    config.option.allure_report_dir = allure_dir

    # ----- Add pytest-html metadata (non-conflicting) -----
    if hasattr(config, "_metadata"):
        config._metadata['Project'] = 'Airline Automation'
        config._metadata['Module'] = 'Booking Engine'
        config._metadata['Tester'] = 'Naveen Kumar'


# -------------------------------
# Logger fixture
# -------------------------------
@pytest.fixture(scope="session")
def logger():
    return LoggerFactory.get_logger("pytest_logger")

@pytest.fixture
def assertions(logger):
    return Assertions(logger)

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
    # Default: returns LangChain conversation messages
    question_chathistory = ironman.get_multiturn_conversation_data(test_data,as_object=True)
    conversation, reference, reference_contexts, synthesizer_name = ironman.get_multiturn_conversation_data(test_data, as_object=False)

    logger.info(question_chathistory)
    logger.info(conversation)
    logger.info(reference)
    logger.info(reference_contexts)
    logger.info(synthesizer_name)

    response_dictionary = ironman.get_rahul_shetty_llm_api_response(question_chathistory)

    # Create multi-turn sample
    user_inputs = []
    for msg in conversation:
        if msg.type == "human":
            user_inputs.append(msg)  # keep HumanMessage
        elif msg.type == "ai":
            user_inputs.append(msg)  # keep AIMessage

    # ---------------------------------------------------------------
    # Build MultiTurnSample
    # ---------------------------------------------------------------
    sample = MultiTurnSample(
        user_input=user_inputs,
        retrieved_contexts=[doc["page_content"] for doc in response_dictionary.get("retrieved_docs", [])],
        response=[response_dictionary.get("answer", "")],
        references=test_data.get("reference", "")
    )
    response = [response_dictionary.get("answer", "")]
    return sample, response, question_chathistory


# Remove unnecessary metadata
def pytest_metadata(metadata):
    metadata.pop('Packages', None)
    metadata.pop('Plugins', None)

def send_email_with_report(logger):

    # Load SMTP details from .env
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT"))
    REPORT_PATH = "testreports/report.html"
    sender = os.getenv("SENDER_EMAIL")

    # Receiver list support
    receivers = os.getenv("EMAIL_RECEIVER_LIST")
    if receivers:
        receivers = [r.strip() for r in receivers.split(",")]
    else:
        receivers = [os.getenv("RECEIVER_EMAIL")]

    subject = os.getenv("EMAIL_SUBJECT", "Pytest Execution Report")
    body = "Hello,<br><br>Please find the attached Pytest execution report.<br><br>Regards,<br>SabreMosaic QA"

    # Prepare email
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = ", ".join(receivers)
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    # Attach HTML report
    with open(REPORT_PATH, "rb") as f:
        part = MIMEApplication(f.read(), Name="report.html")
        part["Content-Disposition"] = 'attachment; filename=\"report.html\"'
        message.attach(part)

    # Try SMTP with and without TLS automatically
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()

            try:
                server.starttls()
                logger.info("🔐 TLS connection established")
            except Exception:
                logger.info("⚠️ TLS not supported, sending without TLS")

            server.sendmail(sender, receivers, message.as_string())

        logger.info("📧 Email sent successfully!")

    except Exception as e:
        logger.info(f"❌ Error sending email: {e}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    logger = LoggerFactory.get_logger("pytest_logger")
    logger.info("pytest execution finished. Sending email report...")
    send_email_with_report(logger)
