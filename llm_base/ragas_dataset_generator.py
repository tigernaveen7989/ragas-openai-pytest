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

# ✅ Load environment variables
load_dotenv()


class RagasDatasetGenerator:
    """
    Utility for generating and uploading RAGAS testsets (single-turn or multi-turn)
    to LangSmith based on feature-specific Word documents.
    """

    def __init__(self, feature_name: str, base_dir: Path = None, dataset_name: str = None, logger=None):
        self.logger = logger
        self.feature_name = feature_name

        self.project_root = self.get_sys_root()

        # Directories
        self.feature_dir = self.project_root / "feature_documents" / feature_name
        self.output_dir = self.project_root / "dataset" / feature_name
        self.output_path = self.output_dir / f"{feature_name.lower()}_dataset.json"

        self.dataset_name = dataset_name or f"ragas_eval_{feature_name.lower()}"
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise EnvironmentError("❌ OPENAI_API_KEY not found. Please set it in .env file.")

        if self.logger:
            self.logger.info(f"📁 Feature Directory: {self.feature_dir}")
            self.logger.info(f"💾 Output Directory: {self.output_dir}")
            self.logger.info(f"📊 Dataset Name: {self.dataset_name}")

    # -------------------------------------------------------------------------
    # ✅ Single-turn Dataset Generator
    # -------------------------------------------------------------------------
    def generate_singleturn_dataset_and_upload(self, testset_size: int):
        """Generates RAGAS single-turn dataset and uploads it to LangSmith."""
        if self.logger:
            self.logger.info(f"🚀 Generating single-turn dataset for feature: {self.feature_name}")

        llm = ChatOpenAI(api_key=self.api_key, model="gpt-4o-mini", temperature=0)
        langchain_llm = LangchainLLMWrapper(llm)
        embedding = OpenAIEmbeddings(api_key=self.api_key)
        generate_embeddings = LangchainEmbeddingsWrapper(embedding)

        loader = DirectoryLoader(
            path=str(self.feature_dir),
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
        )
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"❌ No documents found in {self.feature_dir}")

        generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
        dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        df = dataset.to_pandas()

        os.makedirs(self.output_dir, exist_ok=True)
        df.to_json(self.output_dir / f"singleturn_dataset.json", orient="records", indent=2, force_ascii=False)

        if self.logger:
            self.logger.info(f"💾 Dataset saved at: {self.output_path}")

        self.upload_to_langsmith(self.output_path)
        return str(self.output_path)

    # -------------------------------------------------------------------------
    # ✅ Multi-turn Dataset Generator
    # -------------------------------------------------------------------------
    def generate_multiturn_dataset_and_upload(self, testset_size: int, turns_per_sample: int = 3):
        """
        Generates a multi-turn conversational dataset from feature documents.
        Each sample will include a conversation of N turns between human and AI.
        """
        if self.logger:
            self.logger.info(f"🚀 Generating multi-turn dataset for feature: {self.feature_name}")

        # Initialize models
        llm = ChatOpenAI(api_key=self.api_key, model="gpt-4o-mini", temperature=0.3)
        langchain_llm = LangchainLLMWrapper(llm)
        embedding = OpenAIEmbeddings(api_key=self.api_key)
        generate_embeddings = LangchainEmbeddingsWrapper(embedding)

        # Load feature docs
        loader = DirectoryLoader(
            path=str(self.feature_dir),
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
        )
        docs = loader.load()
        if not docs:
            raise FileNotFoundError(f"❌ No documents found in {self.feature_dir}")

        # Generate base dataset
        generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
        base_dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)
        df = base_dataset.to_pandas()

        # Convert to multi-turn structure
        multiturn_records = []
        for _, row in df.iterrows():
            # Simulate conversation turns
            conversation = []
            question = row["user_input"]
            answer = row["reference"]

            # Generate multiple related questions
            conversation.append({"role": "human", "content": question})
            conversation.append({"role": "ai", "content": answer})

            # Use AI to expand further turns
            for turn in range(1, turns_per_sample):
                follow_up = llm.invoke(
                    f"Given the previous Q&A about '{self.feature_name}', "
                    f"generate a follow-up user question #{turn+1} and a realistic AI response. "
                    f"Keep it short and relevant."
                ).content
                conversation.append({"role": "human", "content": f"Follow-up Q{turn+1}: " + follow_up})
                conversation.append({"role": "ai", "content": f"AI Response {turn+1}: " + follow_up})

            multiturn_records.append({
                "conversation": conversation,
                "reference_contexts": row["reference_contexts"],
                "reference": row["reference"],
                "synthesizer_name": "multi_turn_conversation_generator"
            })

        # Save locally
        os.makedirs(self.output_dir, exist_ok=True)
        multiturn_path = self.output_dir / f"multiturn_dataset.json"

        import json
        with open(multiturn_path, "w", encoding="utf-8") as f:
            json.dump(multiturn_records, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(f"💾 Multi-turn dataset saved at: {multiturn_path}")

        self.upload_to_langsmith(multiturn_path)
        return str(multiturn_path)

    # -------------------------------------------------------------------------
    # ✅ Upload helper
    # -------------------------------------------------------------------------
    def upload_to_langsmith(self, dataset_path: Path):
        """Uploads JSON dataset to LangSmith."""
        dataset = load_dataset("json", data_files=str(dataset_path))
        client = Client()

        try:
            langsmith_dataset = client.read_dataset(self.dataset_name)
            if self.logger:
                self.logger.info(f"✅ Using existing dataset: {self.dataset_name}")
        except Exception:
            langsmith_dataset = client.create_dataset(
                dataset_name=self.dataset_name,
                description="Generated dataset for LLM evaluation",
            )
            if self.logger:
                self.logger.info(f"🆕 Created new dataset: {self.dataset_name}")

        for record in dataset["train"]:
            client.create_example(
                inputs={"conversation": record.get("conversation")},
                outputs={"reference": record.get("reference")},
                dataset_id=langsmith_dataset.id,
            )

        if self.logger:
            self.logger.info(f"✅ Uploaded dataset '{self.dataset_name}' to LangSmith")

    # -------------------------------------------------------------------------
    def get_sys_root(self) -> Path:
        """Finds and returns project root path."""
        return Path(sys.path[0]).resolve()
