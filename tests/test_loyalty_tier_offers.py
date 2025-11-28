import allure
import pytest
from llm_base.ragas_metrics_evaluator import MetricsEvaluator
from utilities.ironman import IronMan
from utilities.assertions import Assertions


@allure.suite("Loyalty Tier Offers Evaluation Suite")
@allure.feature("Loyalty Tier Offers")
class TestLoyaltyTierOffers:
    feature_name = "loyalty_tier_offers"

    @allure.story("Faithfulness Evaluation")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.description(
        "Validates that the model response remains faithful to the reference context for loyalty tier offers.")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_faithfulness(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate faithfulness score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_faithfulness_score(get_singleturn_data)
        logger.info(f"Faithfulness Score: {score}")

        # Validation threshold
        assertions.assert_faithfulness(score)

    # -------------------------------------------------------------------------

    @allure.story("Context Precision Evaluation")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.description(
        "Validates that retrieved contexts are relevant to the user query (Context Precision)."
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_context_precision(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate Context Precision score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_context_precision_score(get_singleturn_data)
        logger.info(f"Context Precision Score: {score}")

        assertions.assert_context_precision(score)

    # -------------------------------------------------------------------------
    @allure.story("Context Recall Evaluation")
    @allure.severity(allure.severity_level.MINOR)
    @allure.description(
        "Validates that all relevant pieces of context were successfully retrieved (Context Recall)."
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_context_recall(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate Context Recall score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_context_recall_score(get_singleturn_data)
        logger.info(f"Context Recall Score: {score}")

        assertions.assert_context_recall(score)

    # -------------------------------------------------------------------------
    @allure.story("Answer Relavancy Evaluation")
    @allure.severity(allure.severity_level.BLOCKER)
    @allure.description(
        "Validates that the model response is relevant to the user query (Answer Relevancy)."
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_answer_relevancy(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate Answer Relevancy score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_answer_relevancy_score(get_singleturn_data)
        logger.info(f"Answer Relevancy Score: {score}")

        assertions.assert_answer_relevancy(score)

    # -------------------------------------------------------------------------
    @allure.story("Factual Correctness Evaluation")
    @allure.severity(allure.severity_level.TRIVIAL)
    @allure.description(
        "Validates that the generated answer is factually correct with respect to the reference answer."
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_factual_correctness(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate Factual Correctness score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_factual_correctness_score(get_singleturn_data)
        logger.info(f"Factual Correctness Score: {score}")

        assertions.assert_factual_correctness(score)

    # -------------------------------------------------------------------------

    @allure.story("Rubric Evaluation")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.description(
        "Validates that the model response meets predefined quality rubric such as accuracy, completeness, and coherence."
    )
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_singleturn_data",
        IronMan.load_test_data("loyalty_tier_offers"),
        indirect=True
    )
    async def test_rubric_score(self, get_llm_wrapper, get_singleturn_data, logger, assertions):
        """
        Test to validate Rubric Score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_rubric_score(get_singleturn_data)
        logger.info(f"Rubric Score: {score}")

        assertions.assert_rubric(score)
