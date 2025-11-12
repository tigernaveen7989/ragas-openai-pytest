import os
import sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

# ✅ Load environment variables from .env (project root)
load_dotenv()


class RagasDatasetGenerator:
    """
    Reusable utility for generating and uploading RAGAS datasets to LangSmith.
    Automatically resolves system directories (no hardcoded paths).
    """

    def __init__(self, feature_name: str, base_dir: Path = None, dataset_name: str = None, logger=None):
        self.logger = logger
        self.feature_name = feature_name

        self.project_root = self.get_sys_root()

        # ✅ Construct dynamic feature & dataset directories
        self.feature_dir = self.project_root / "feature_documents" / feature_name
        self.output_dir = self.project_root / "dataset" / feature_name
        self.output_path = self.output_dir / f"{feature_name.lower()}_dataset.json"

        self.dataset_name = dataset_name or f"ragas_eval_{feature_name.lower()}"
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise EnvironmentError("❌ OPENAI_API_KEY not found. Make sure it's set in your .env file.")

        if self.logger:
            self.logger.info(f"📁 Feature Directory: {self.feature_dir}")
            self.logger.info(f"💾 Output Directory: {self.output_dir}")
            self.logger.info(f"📊 Dataset Name: {self.dataset_name}")

    def generate_and_upload(self, testset_size: int):
        """Generates RAGAS testset and uploads it to LangSmith."""

        if self.logger:
            self.logger.info(f"🚀 Generating dataset for feature: {self.feature_name}")

        # ✅ Initialize LLM and embeddings
        llm = ChatOpenAI(api_key=self.api_key, model="gpt-4o-mini", temperature=0)
        langchain_llm = LangchainLLMWrapper(llm)
        embedding = OpenAIEmbeddings(api_key=self.api_key)
        generate_embeddings = LangchainEmbeddingsWrapper(embedding)

        # ✅ Load feature documents dynamically
        loader = DirectoryLoader(
            path=str(self.feature_dir),
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
        )
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"❌ No documents found in {self.feature_dir}")
        if self.logger:
            self.logger.info(f"✅ Loaded {len(docs)} documents for {self.feature_name}")

        # ✅ Generate dataset
        generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
        data_set = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        df = data_set.to_pandas()

        # ✅ Save dataset locally
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_json(self.output_path, orient="records", indent=2, force_ascii=False)
        if self.logger:
            self.logger.info(f"💾 Dataset saved at: {self.output_path}")

        # ✅ Upload to LangSmith
        dataset = load_dataset("json", data_files=str(self.output_path))
        client = Client()
        if self.logger:
            self.logger.info("✅ Connected to LangSmith client successfully.")

        # 🔄 Try delete if exists
        try:
            langsmith_dataset = client.read_dataset(self.dataset_name)
            self.logger(f"✅ Using existing dataset: {langsmith_dataset}")
        except Exception:
            # 🆕 Create fresh one
            langsmith_dataset = client.create_dataset(
                dataset_name=self.dataset_name,
                description="Regenerated dataset for LLM evaluation"
            )
            self.logger(f"✅ Created new dataset: {langsmith_dataset}")

        for record in dataset["train"]:
            client.create_example(
                inputs={
                    "user_input": record.get("user_input"),
                    "reference_contexts": record.get("reference_contexts"),
                    "synthesizer_name": record.get("synthesizer_name"),
                },
                outputs={"reference": record.get("reference")},
                dataset_id=langsmith_dataset.id,
            )

        if self.logger:
            self.logger.info(f"✅ Uploaded all records to LangSmith dataset '{self.dataset_name}'")
            self.logger.info("🌐 View dataset at: https://smith.langchain.com/datasets")

        return str(self.output_path)

    def get_sys_root(self) -> Path:
        """
        Finds and returns the project root path.

        Returns:
            Path: The resolved absolute path of the project's root directory.
        """
        # Typically, the first entry in sys.path is the project root
        return Path(sys.path[0]).resolve()

