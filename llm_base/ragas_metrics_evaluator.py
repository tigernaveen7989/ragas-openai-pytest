import os

from langchain_core.prompt_values import ChatPromptValue
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision, AnswerRelevancy, RubricsScore, \
    FactualCorrectness
import allure
import json
from ragas.metrics import AspectCritic
from ragas.dataset_schema import EvaluationDataset
from ragas.messages import HumanMessage
from ragas import evaluate
from langchain_core.messages import HumanMessage, SystemMessage
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample


class MetricsEvaluator:
    """
        A reusable helper class to compute Faithfulness scores
        for LLM test samples.
        """

    def __init__(self, get_llm_wrapper):
        self.get_llm_wrapper = get_llm_wrapper

    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------

    async def get_rubric_score(self, sample):
        """
        Compute the Rubric Score for a given test sample.
        Works for both Single-turn and Multi-turn conversations.
        """
        with allure.step("Calculate Rubric Score"):
            # Attach details for traceability
            allure.attach(str(sample.user_input), "User Input", allure.attachment_type.TEXT)
            if hasattr(sample, "retrieved_contexts"):
                allure.attach(json.dumps(sample.retrieved_contexts, indent=2), "Retrieved Contexts",
                              allure.attachment_type.TEXT)
            if hasattr(sample, "response"):
                allure.attach(str(sample.response), "Model Response", allure.attachment_type.TEXT)
            if hasattr(sample, "reference"):
                allure.attach(str(sample.reference), "Reference Answer", allure.attachment_type.TEXT)

            # Define rubric scale
            rubrics = {
                "score1_description": "Response/conversation is incorrect or irrelevant.",
                "score2_description": "Partially correct but misses key details or context.",
                "score3_description": "Generally correct but lacks clarity or completeness.",
                "score4_description": "Mostly accurate and context-aware with minor issues.",
                "score5_description": "Fully accurate, clear, complete, and context-aware."
            }

            allure.attach(json.dumps(rubrics, indent=2), name="Rubric Definitions",
                          attachment_type=allure.attachment_type.JSON)

            # Initialize RubricsScore metric
            metric = RubricsScore(rubrics=rubrics, llm=self.get_llm_wrapper)

            # Detect sample type and compute score
            if isinstance(sample, SingleTurnSample):
                score = await metric.single_turn_ascore(sample)
            elif isinstance(sample, MultiTurnSample):
                score = await metric.multi_turn_ascore(sample)
            else:
                raise TypeError(f"Unsupported sample type: {type(sample).__name__}")

            # Attach score for reporting
            allure.attach(str(score), "Rubric Score", allure.attachment_type.TEXT)

        with allure.step(f"Rubric Score: {score}"):
            pass

        return score

    # ----------------------------------------------------------------------
    async def get_multiturn_faithfulness_score(self, sample):
        """
        Compute multi-turn faithfulness by computing per-turn faithfulness
        using single_turn_ascore(), since RAGAS has no multi-turn API for this metric.
        """

        with allure.step("Calculate Multi-Turn Faithfulness Score"):

            # Attach full conversation
            conversation_serialized = [
                {"role": msg.type, "content": msg.content}
                for msg in sample.user_input
            ]
            allure.attach(
                json.dumps(conversation_serialized, indent=2, ensure_ascii=False),
                name="Conversation (User Input: Multi-Turn)",
                attachment_type=allure.attachment_type.TEXT
            )

            # Track per-turn scores
            turn_scores = []

            # Extract messages in pairs (human → ai)
            messages = sample.user_input

            # retrieved_contexts is a list per turn
            contexts = getattr(sample, "retrieved_contexts", [])

            # Instantiate single-turn faithfulness
            metric = Faithfulness(llm=self.get_llm_wrapper)

            turn_index = 0

            for i in range(0, len(messages), 2):

                if i + 1 >= len(messages):
                    break  # uneven turns, skip

                user_msg = messages[i]
                ai_msg = messages[i + 1]

                # Build a temporary single-turn sample
                single_turn_sample = type("SingleTurnSample", (), {})()

                single_turn_sample.user_input = user_msg.content
                single_turn_sample.response = ai_msg.content

                # Map context (if available)
                if turn_index < len(contexts):
                    single_turn_sample.retrieved_contexts = contexts[turn_index]
                else:
                    single_turn_sample.retrieved_contexts = []

                # Compute single-turn faithfulness
                score = await metric.single_turn_ascore(single_turn_sample)
                turn_scores.append(score)

                turn_index += 1

            # Final score = average of all turn scores
            final_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0

            allure.attach(
                json.dumps(turn_scores, indent=2),
                name="Per-Turn Faithfulness Scores",
                attachment_type=allure.attachment_type.TEXT,
            )

            allure.attach(
                str(final_score),
                name="Multi-Turn Faithfulness Score",
                attachment_type=allure.attachment_type.TEXT,
            )

        with allure.step(f"Multi-Turn Faithfulness Score: {final_score}"):
            pass

        return final_score

    # ----------------------------------------------------------------------
    async def get_multiturn_answer_relevance_score(self, sample):
        """
        Compute the Answer Relevance score for a multi-turn test sample
        and log inputs in Allure.
        """

        with allure.step("Calculate Multi-Turn Answer Relevance Score"):

            # Attach user_input (multi-turn conversation)
            try:
                conversation_serialized = [
                    {"role": msg.type, "content": msg.content}
                    for msg in sample.user_input
                ]

                allure.attach(
                    json.dumps(conversation_serialized, indent=2, ensure_ascii=False),
                    name="Conversation (User Input: Multi-Turn)",
                    attachment_type=allure.attachment_type.TEXT
                )
            except:
                allure.attach(
                    str(sample.user_input),
                    name="Raw Conversation Data",
                    attachment_type=allure.attachment_type.TEXT
                )

            # Attach retrieved contexts if present
            if hasattr(sample, "retrieved_contexts"):
                allure.attach(
                    json.dumps(sample.retrieved_contexts, indent=2, ensure_ascii=False),
                    name="Retrieved Contexts",
                    attachment_type=allure.attachment_type.TEXT
                )

            # Instantiate Answer Relevance metric
            self.metric = AnswerRelevancy(llm=self.get_llm_wrapper)

            # Calculate multi-turn score
            score = await self.metric.multi_turn_ascore(sample)

            # Attach the score
            allure.attach(
                str(score),
                name="Answer Relevance Score (Multi-Turn)",
                attachment_type=allure.attachment_type.TEXT
            )

        with allure.step(f"Multi-Turn Answer Relevance Score: {score}"):
            pass

        return score

    async def get_aspect_critic(self, sample):
        """
        Compute the AspectCritic score for a given test sample
        based on the provided critique aspect (e.g., correctness, clarity).
        """
        with allure.step(f"Calculate AspectCritic (Aspect Critic:)"):
            # Attach user input
            allure.attach(
                str(sample.user_input),
                name="User Input",
                attachment_type=allure.attachment_type.TEXT
            )

            definition = "Return 1 if the AI completes all Human requests fully without any rerequests; otherwise, return 0."

            aspect_critic = AspectCritic(
                name="forgetfulness_aspect_critic",
                definition=definition,
                llm=self.get_llm_wrapper,
            )

            result = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[aspect_critic],
            )

            # Attach the metric result
            allure.attach(
                str(result),
                name=f"AspectCritic Result",
                attachment_type=allure.attachment_type.TEXT
            )

        with allure.step(f"AspectCritic Result: {result}"):
            pass

        return result

    # -----------------------------------------------------------

    async def get_top_adherence_score(self, sample, response):
        """
                    Compute Top Adherence Score in a single function.
                    Evaluates 6 adherence dimensions using the LLM and returns a weighted score.
                    LLM instance is created inside this function.
                    """
        with allure.step("Calculate Top Adherence Score"):

            # Attach input & response
            allure.attach(
                str(sample.user_input),
                name="User Input",
                attachment_type=allure.attachment_type.TEXT
            )
            allure.attach(
                str(response),
                name="Chatbot Response",
                attachment_type=allure.attachment_type.TEXT
            )

            # ------------------------------------------------------
            # 🔥 1. Get the LLM Instance
            # ------------------------------------------------------
            llm_client = self.get_llm_wrapper

            # ------------------------------------------------------
            # 🔥 2. Scoring Prompt
            # ------------------------------------------------------
            prompt = f"""
            Evaluate the assistant response across 6 adherence dimensions (scale 0–1).
            Return ONLY a JSON dictionary.

            User Input:
            {sample.user_input}

            Assistant Response:
            {response}

            Rate each of the following between 0 and 1:

            1. instruction_adherence
            2. constraint_satisfaction
            3. hallucination_avoidance
            4. relevance_score
            5. format_score
            6. safety_policy_adherence

            Return JSON ONLY:
            {{
                "instruction_adherence": <float>,
                "constraint_satisfaction": <float>,
                "hallucination_avoidance": <float>,
                "relevance_score": <float>,
                "format_score": <float>,
                "safety_policy_adherence": <float>
            }}
            """

            # ------------------------------------------------------
            # 🔥 3. LLM Call (Fixed)
            # ------------------------------------------------------
            prompt_value = ChatPromptValue(messages=[
                SystemMessage(content="You are an evaluator. Return only JSON."),
                HumanMessage(content=prompt)
            ])

            llm_response = await llm_client.agenerate_text(prompt_value)
            # ------------------------------------------------------
            # 🔥 4. Extract CLEAN JSON String
            # ------------------------------------------------------
            gen = llm_response.generations[0][0]  # ChatGeneration
            raw_text = gen.text or gen.message.content

            # Remove ```json ``` code fences
            clean_text = (
                raw_text.replace("```json", "")
                .replace("```", "")
                .strip()
            )
            # ------------------------------------------------------
            # 🔥 4. Parse JSON Output
            # ------------------------------------------------------
            try:
                scores = json.loads(clean_text)
            except Exception as e:
                raise ValueError(
                    "❌ LLM did not return valid JSON.\n"
                    f"Raw text:\n{raw_text}\n"
                    f"Cleaned:\n{clean_text}"
                ) from e

            # ------------------------------------------------------
            # 🔥 5. Weighted Score Calculation
            # ------------------------------------------------------
            weights = {
                "instruction_adherence": 0.25,
                "constraint_satisfaction": 0.20,
                "hallucination_avoidance": 0.20,
                "relevance_score": 0.15,
                "format_score": 0.10,
                "safety_policy_adherence": 0.10,
            }

            final_score = sum(scores[k] * weights[k] for k in weights)
            final_score = round(final_score, 4)

            # Attach scores
            allure.attach(str(scores),
                          "Individual Adherence Scores",
                          allure.attachment_type.TEXT)

            allure.attach(str(final_score),
                          "Final Top Adherence Score",
                          allure.attachment_type.TEXT)

        with allure.step(f"Top Adherence Score: {final_score}"):
            pass

        return final_score

