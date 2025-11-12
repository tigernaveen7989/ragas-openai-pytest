import pytest
from llm_base.ragas_dataset_generator import RagasDatasetGenerator
from pathlib import Path

class TestRagasDatasetGenerator:

    def test_generate_dynamic_dataset(self, logger):
        """Reusable dataset generation test using system paths."""
        feature_name = "cancel_booking"  # only this changes

        generator = RagasDatasetGenerator(
            feature_name=feature_name,
            base_dir=Path.cwd(),  # dynamically uses current working directory
            logger=logger
        )

        output_path = generator.generate_and_upload(testset_size=5)
        logger.info(f"✅ Dataset generated and uploaded successfully: {output_path}")
