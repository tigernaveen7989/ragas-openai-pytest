import allure
import pytest

from llm_base.ragas_metrics_evaluator import MetricsEvaluator
from utilities.assertions import Assertions
from utilities.ironman import IronMan


@allure.suite("Flight Booking Evaluation Suite")
@allure.feature("Flight Booking")
class TestFlightBooking:
    feature_name = "flight_booking"
    assertions = Assertions()

    @allure.description(
        "Validates that the model response remains faithful to the reference context for flight booking")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("get_singleturn_data", IronMan.load_test_data(feature_name), indirect=True)
    async def test_faithfulness(self, get_llm_wrapper, get_singleturn_data, logger):
        """
        Test to validate faithfulness score using reusable helper class.
        """
        evaluator = MetricsEvaluator(get_llm_wrapper)
        score = await evaluator.get_faithfulness_score(get_singleturn_data)
        logger.info(f"Faithfulness Score: {score}")

        # Validation threshold
        self.assertions.assert_faithfulness(score)
    # -------------------------------------------------------------------------