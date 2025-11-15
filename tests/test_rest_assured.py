import allure
import pytest

from llm_base.ragas_metrics_evaluator import MetricsEvaluator
from utilities.assertions import Assertions
from utilities.ironman import IronMan


@allure.suite("Rest Assured Evaluation Suite")
@allure.feature("Rest Assured")
class TestRestAssured:
    feature_name = "rest_assured"
    assertions = Assertions()

    @allure.story("Aspect Critic Evaluation")
    @allure.description(
        "Validates that the model response remains aspect critic to the reference context for rest assured")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_multiturn_data", IronMan.load_test_data(feature_name, "multiturn"), indirect=True)
    async def test_aspect_critic(self, get_llm_wrapper, get_multiturn_data, logger):
        """
        Test to validate aspect critic score using reusable helper class.
        """
        logger.info(get_multiturn_data)
        sample, response = get_multiturn_data
        evaluator = MetricsEvaluator(get_llm_wrapper)
        result = await evaluator.get_aspect_critic(sample=sample)
        logger.info(f"Aspect Critic Result: {result}")
        self.assertions.assert_aspect_critic(result, threshold=0.7)

    # -------------------------------------------------------------------------

    @allure.story("Top Adherence Evaluation")
    @allure.description(
        "Validates that the model response remains top adherence to the reference context for rest assured")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_multiturn_data", IronMan.load_test_data(feature_name, "multiturn"), indirect=True)
    async def test_top_adherence_score(self, get_llm_wrapper, get_multiturn_data, logger):
        """
        Test to validate aspect critic score using reusable helper class.
        """
        sample, response = get_multiturn_data
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_top_adherence_score(sample=sample, response=response)
        logger.info(f"Top Adherence Score: {score}")
        self.assertions.assert_top_adherence(score=score, threshold=0.8)
    # -------------------------------------------------------------------------

    @allure.story("Rubric Evaluation")
    @allure.description(
        "Validates that the model response remains rubric to the reference context for rest assured")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_multiturn_data", IronMan.load_test_data(feature_name, "multiturn"), indirect=True)
    async def test_rubric_score(self, get_llm_wrapper, get_multiturn_data, logger):
        """
        Test to validate Rubric Score using reusable helper class.
        """
        sample, response = get_multiturn_data
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_rubric_score(sample)
        logger.info(f"Rubric Score: {score}")

        self.assertions.assert_rubric(score)
