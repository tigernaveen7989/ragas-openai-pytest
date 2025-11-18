import allure

from utilities.logger import LoggerFactory


class Assertions:
    def __init__(self, logger):
        self.logger = logger

    def assert_context_precision(self, score: float, threshold: float = 0.7):
        """
        Validate that the Context Precision score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Context Precision ≥ {threshold}"):
            assert score > threshold, f"❌ Context Precision too low: {score}. Expected > {threshold}"
            with allure.step(f"Context Precision Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Context Precision Validation Passed ({score})"):
                pass

    def assert_context_recall(self, score: float, threshold: float = 0.7):
        """
        Validate that the Context Recall score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Context Recall ≥ {threshold}"):
            assert score > threshold, f"❌ Context Recall too low: {score}. Expected > {threshold}"
            with allure.step(f"Context Recall Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Context Recall Validation Passed ({score})"):
                pass

    def assert_answer_relevancy(self, score: float, threshold: float = 0.7):
        """
        Validate that the Answer Relevancy score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Answer Relevancy ≥ {threshold}"):
            assert score > threshold, f"❌ Answer Relevancy too low: {score}. Expected > {threshold}"
            with allure.step(f"Answer Relevancy Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Answer Relevancy Validation Passed ({score})"):
                pass

    def assert_factual_correctness(self, score: float, threshold: float = 0.7):
        """
        Validate that the Factual Correctness score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Factual Correctness ≥ {threshold}"):
            assert score > threshold, f"❌ Factual Correctness too low: {score}. Expected > {threshold}"
            with allure.step(f"Factual Correctness Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Factual Correctness Validation Passed ({score})"):
                pass

    def assert_faithfulness(self, score: float, threshold: float = 0.7):
        """
        Validate that the Faithfulness score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Faithfulness Score ≥ {threshold}"):
            assert score > threshold, f"❌ Faithfulness score too low: {score}. Expected > {threshold}"
            with allure.step(f"Faithfulness Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Faithfulness Score Validation Passed ({score})"):
                pass

    def assert_rubric(self, score: float, threshold: float = 3):
        """
        Validate that the Rubrics score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Rubrics Score ≥ {threshold}"):
            assert score >= threshold, f"❌ Rubrics too low: {score}. Expected > {threshold}"
            with allure.step(f"Rubrics Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Rubrics Validation Passed ({score})"):
                pass

    def assert_aspect_critic(self, result, threshold: float = 0.7):
        """
        Validate that the AspectCritic score meets the minimum threshold.
        Result comes as {'aspectname_aspect_critic': score}.
        """
        score_list = result.scores
        score = score_list[0] if score_list else None
        score = score['forgetfulness_aspect_critic']

        if score is None:
            raise ValueError("No score found in EvaluationResult")

        with allure.step(f"Validate AspectCritic Score ≥ {threshold}"):
            assert score > threshold, (
                f"❌ AspectCritic score too low: {score}. "
                f"Expected > {threshold}. Aspect: {score}"
            )

            with allure.step(f"Aspect: {score}"):
                pass
            with allure.step(f"AspectCritic Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ AspectCritic Score Validation Passed ({score})"):
                pass

        return score

    def assert_top_adherence(self, score: float, threshold: float = 0.8):
        """
        Validate that the Top Adherence score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Top Adherence Score ≥ {threshold}"):
            assert score >= threshold, f"❌ Top Adherence too low: {score}. Expected > {threshold}"
            with allure.step(f"Top Adherence Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Top Adherence Validation Passed ({score})"):
                pass

    def assert_conversational_memory(self, score: float, threshold: float = 0.8):
        """
        Validate that the Conversational Memory score meets the minimum threshold.
        """
        self.logger.info(f"Expected Threshold: {threshold}, Score: {score}")
        with allure.step(f"Validate Conversational Memory Score ≥ {threshold}"):
            assert score >= threshold, f"❌ Conversational Memory too low: {score}. Expected > {threshold}"
            with allure.step(f"Conversational Memory Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Conversational Memory Validation Passed ({score})"):
                pass