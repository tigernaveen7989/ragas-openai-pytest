from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy, RubricsScore, FactualCorrectness
import allure
import json

class MetricsEvaluator:
    """
        A reusable helper class to compute Faithfulness scores
        for LLM test samples.
        """

    def __init__(self, get_llm_wrapper):
        self.get_llm_wrapper = get_llm_wrapper

    async def get_faithfulness_score(self, sample):
        """
                Compute the Faithfulness score for a given test sample
                and log details in Allure.
                """
        with allure.step("Calculate Faithfulness Score"):
            # Attach input details for traceability
            allure.attach(
                str(sample.user_input),
                name="User Input",
                attachment_type=allure.attachment_type.TEXT
            )
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                name="Retrieved Contexts",
                attachment_type=allure.attachment_type.TEXT
            )
            allure.attach(
                str(sample.response),
                name="Reference Response",
                attachment_type=allure.attachment_type.TEXT
            )

            # Metric computation step
            self.metric = Faithfulness(llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            # Attach the result score
            allure.attach(
                str(score),
                name="Faithfulness Score",
                attachment_type=allure.attachment_type.TEXT
            )

        with allure.step(f"Faithfulness Score: {score}"):
            pass

        return score

    # ----------------------------------------------------------------------
    async def get_context_precision_score(self, sample):
        """
        Compute the Context Precision score for a given test sample.
        """
        with allure.step("Calculate Context Precision Score"):
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                "Retrieved Contexts",
                allure.attachment_type.TEXT
            )
            allure.attach(str(sample.response), "Reference Response", allure.attachment_type.TEXT)

            self.metric = ContextPrecision(llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            allure.attach(str(score), "Context Precision Score", allure.attachment_type.TEXT)

        with allure.step(f"Context Precision Score: {score}"):
            pass
        return score

    # ----------------------------------------------------------------------
    async def get_context_recall_score(self, sample):
        """
        Compute the Context Recall score for a given test sample.
        """
        with allure.step("Calculate Context Recall Score"):
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                "Retrieved Contexts",
                allure.attachment_type.TEXT
            )
            allure.attach(str(sample.response), "Reference Response", allure.attachment_type.TEXT)

            self.metric = ContextRecall(llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            allure.attach(str(score), "Context Recall Score", allure.attachment_type.TEXT)

        with allure.step(f"Context Recall Score: {score}"):
            pass
        return score

    # ----------------------------------------------------------------------
    async def get_answer_relevancy_score(self, sample):
        """
        Compute the Answer Relevancy score for a given test sample.
        """
        with allure.step("Calculate Answer Relevancy Score"):
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                "Retrieved Contexts",
                allure.attachment_type.TEXT
            )
            allure.attach(str(sample.response), "Reference Response", allure.attachment_type.TEXT)

            self.metric = AnswerRelevancy(llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            allure.attach(str(score), "Answer Relevancy Score", allure.attachment_type.TEXT)

        with allure.step(f"Answer Relevancy Score: {score}"):
            pass
        return score

    # ----------------------------------------------------------------------
    async def get_factual_correctness_score(self, sample):
        """
        Compute the Factual Correctness score for a given test sample.
        """
        with allure.step("Calculate Factual Correctness Score"):
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                "Retrieved Contexts",
                allure.attachment_type.TEXT
            )
            allure.attach(str(sample.response), "Reference Response", allure.attachment_type.TEXT)

            self.metric = FactualCorrectness(llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            allure.attach(str(score), "Factual Correctness Score", allure.attachment_type.TEXT)

        with allure.step(f"Factual Correctness Score: {score}"):
            pass
        return score

    async def get_rubric_score(self, sample):
        """
        Compute the Rubric Score for a given test sample.
        """
        with allure.step("Calculate Rubric Score"):
            # Attach all relevant test inputs for traceability
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            allure.attach(
                json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                "Retrieved Contexts",
                allure.attachment_type.TEXT
            )
            allure.attach(str(sample.response), "Model Response", allure.attachment_type.TEXT)
            allure.attach(str(sample.reference), "Reference Answer", allure.attachment_type.TEXT)

            # Define rubric scale and qualitative meaning
            rubrics = {
                "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
                "score2_description": "The response partially matches the ground truth but includes significant errors or omissions.",
                "score3_description": "The response generally aligns with the ground truth but lacks detail or clarity.",
                "score4_description": "The response is mostly accurate and well-aligned, with only minor issues.",
                "score5_description": "The response is fully accurate, clear, complete, and matches the ground truth perfectly.",
            }

            allure.attach(
                json.dumps(rubrics, indent=2, ensure_ascii=False),
                name="Rubric Definitions",
                attachment_type=allure.attachment_type.JSON)

            # Initialize and compute the metric
            self.metric = RubricsScore(rubrics=rubrics, llm=self.get_llm_wrapper)
            score = await self.metric.single_turn_ascore(sample)

            # Attach the score for reporting
            allure.attach(str(score), "Rubric Score", allure.attachment_type.TEXT)

        with allure.step(f"Rubric Score: {score}"):
            pass

        return score