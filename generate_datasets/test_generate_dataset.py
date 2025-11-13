import pytest
from llm_base.ragas_dataset_generator import RagasDatasetGenerator
from pathlib import Path

class TestRagasDatasetGenerator:

    def test_generate_singleturn_dataset(self, logger):
        """Reusable dataset generation test using system paths."""
        feature_name = "cancel_booking"  # only this changes

        generator = RagasDatasetGenerator(
            feature_name=feature_name,
            base_dir=Path.cwd(),  # dynamically uses current working directory
            logger=logger
        )

        output_path = generator.generate_singleturn_dataset_and_upload(testset_size=5)
        logger.info(f"✅ Dataset generated and uploaded successfully: {output_path}")


    def test_generate_multiturn_dataset(self, logger):
        """Reusable dataset generation test using system paths."""
        feature_name = "cancel_booking"  # only this changes

        generator = RagasDatasetGenerator(
            feature_name=feature_name,
            base_dir=Path.cwd(),  # dynamically uses current working directory
            logger=logger
        )

        output_path = generator.generate_multiturn_dataset_and_upload(testset_size=5)
        logger.info(f"✅ Dataset generated and uploaded successfully: {output_path}")
