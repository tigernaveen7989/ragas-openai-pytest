import allure

class Assertions:

    def assert_context_precision(score: float, threshold: float = 0.7):
        """
        Validate that the Context Precision score meets the minimum threshold.
        """
        with allure.step(f"Validate Context Precision ≥ {threshold}"):
            assert score > threshold, f"❌ Context Precision too low: {score}. Expected > {threshold}"
            with allure.step(f"Context Precision Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Context Precision Validation Passed ({score})"):
                pass

    def assert_context_recall(score: float, threshold: float = 0.7):
        """
        Validate that the Context Recall score meets the minimum threshold.
        """
        with allure.step(f"Validate Context Recall ≥ {threshold}"):
            assert score > threshold, f"❌ Context Recall too low: {score}. Expected > {threshold}"
            with allure.step(f"Context Recall Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Context Recall Validation Passed ({score})"):
                pass

    def assert_answer_relevancy(score: float, threshold: float = 0.7):
        """
        Validate that the Answer Relevancy score meets the minimum threshold.
        """
        with allure.step(f"Validate Answer Relevancy ≥ {threshold}"):
            assert score > threshold, f"❌ Answer Relevancy too low: {score}. Expected > {threshold}"
            with allure.step(f"Answer Relevancy Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Answer Relevancy Validation Passed ({score})"):
                pass

    def assert_factual_correctness(score: float, threshold: float = 0.7):
        """
        Validate that the Factual Correctness score meets the minimum threshold.
        """
        with allure.step(f"Validate Factual Correctness ≥ {threshold}"):
            assert score > threshold, f"❌ Factual Correctness too low: {score}. Expected > {threshold}"
            with allure.step(f"Factual Correctness Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Factual Correctness Validation Passed ({score})"):
                pass

    def assert_faithfulness(score: float, threshold: float = 0.7):
        """
        Validate that the Faithfulness score meets the minimum threshold.
        """
        with allure.step(f"Validate Faithfulness Score ≥ {threshold}"):
            assert score > threshold, f"❌ Faithfulness score too low: {score}. Expected > {threshold}"
            with allure.step(f"Faithfulness Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Faithfulness Score Validation Passed ({score})"):
                pass

    def assert_rubric(score: float, threshold: float = 3):
        """
        Validate that the Rubrics score meets the minimum threshold.
        """
        with allure.step(f"Validate Rubrics Score ≥ {threshold}"):
            assert score >= threshold, f"❌ Rubrics too low: {score}. Expected > {threshold}"
            with allure.step(f"Rubrics Score: {score}"):
                pass
            with allure.step(f"Expected Threshold: {threshold}"):
                pass
            with allure.step(f"✅ Rubrics Validation Passed ({score})"):
                pass
