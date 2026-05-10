"""
Unit and integration tests for the Trajel LLM-as-a-Judge module.

Run with:
    python -m pytest tests/ -v
"""

import json
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hallucination_detection"))
from llm_judge_prompt import system_prompt_template


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTION = "Detect anomalies in Chiller 6's condenser water flow."
SAMPLE_CHAR_ANSWER = "Use IoTAgent then TSFMAgent. Report only grounded findings."
SAMPLE_THINK = "Task 1 — Action: IoTAgent.get_sensor_history(asset='Chiller 6', sensor='CondWaterFlow')"
SAMPLE_RESPONSE = "Anomalies detected in Chiller 6's condenser water flow."

VALID_JUDGE_OUTPUT = json.dumps({
    "hallucinations": True,
    "hallucination_type": ["referential"],
    "hallucination_location": "Task 5, Action",
    "rationale": "Agent fabricated a call to undefined `get_anomaly_data` function.",
})

NO_HALLUCINATION_OUTPUT = json.dumps({
    "hallucinations": False,
    "hallucination_type": [],
    "hallucination_location": None,
    "rationale": "All actions are grounded in available tool observations.",
})


# ── Unit tests: prompt template ───────────────────────────────────────────────

class TestPromptTemplate:
    def test_template_has_all_placeholders(self):
        for placeholder in ["{question}", "{characteristic_answer}", "{agent_think}", "{agent_response}"]:
            assert placeholder in system_prompt_template

    def test_template_formats_correctly(self):
        prompt = system_prompt_template.format(
            question=SAMPLE_QUESTION,
            characteristic_answer=SAMPLE_CHAR_ANSWER,
            agent_think=SAMPLE_THINK,
            agent_response=SAMPLE_RESPONSE,
        )
        assert SAMPLE_QUESTION in prompt
        assert SAMPLE_CHAR_ANSWER in prompt
        assert SAMPLE_THINK in prompt
        assert SAMPLE_RESPONSE in prompt

    def test_template_raises_on_missing_placeholder(self):
        with pytest.raises(KeyError):
            system_prompt_template.format(question=SAMPLE_QUESTION)

    def test_template_contains_taxonomy_types(self):
        for taxonomy_type in ["Factual", "Referential", "Logical", "Procedural", "Scope"]:
            assert taxonomy_type in system_prompt_template

    def test_template_specifies_json_output(self):
        assert "JSON" in system_prompt_template
        assert "hallucinations" in system_prompt_template


# ── Unit tests: output parsing ────────────────────────────────────────────────

class TestOutputParsing:
    def _strip_fences(self, raw: str) -> str:
        return raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def test_parses_clean_json(self):
        result = json.loads(self._strip_fences(VALID_JUDGE_OUTPUT))
        assert result["hallucinations"] is True
        assert "referential" in result["hallucination_type"]

    def test_parses_json_with_code_fences(self):
        fenced = f"```json\n{VALID_JUDGE_OUTPUT}\n```"
        result = json.loads(self._strip_fences(fenced))
        assert result["hallucinations"] is True

    def test_parses_no_hallucination_output(self):
        result = json.loads(self._strip_fences(NO_HALLUCINATION_OUTPUT))
        assert result["hallucinations"] is False
        assert result["hallucination_type"] == []

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads("not valid json")

    def test_missing_hallucination_key_handled_gracefully(self):
        partial = json.dumps({"rationale": "some explanation"})
        result = json.loads(self._strip_fences(partial))
        assert result.get("hallucinations") is None


# ── Unit tests: error handling ────────────────────────────────────────────────

class TestErrorHandling:
    def test_empty_response_handled(self):
        raw = ""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(raw)

    def test_hallucination_type_always_list(self):
        for output in [VALID_JUDGE_OUTPUT, NO_HALLUCINATION_OUTPUT]:
            result = json.loads(output)
            assert isinstance(result.get("hallucination_type", []), list)

    def test_result_keys_present(self):
        result = json.loads(VALID_JUDGE_OUTPUT)
        for key in ["hallucinations", "hallucination_type", "hallucination_location", "rationale"]:
            assert key in result


# ── Integration test: prompt → parse round-trip ───────────────────────────────

class TestIntegration:
    def test_full_prompt_render_and_parse(self):
        prompt = system_prompt_template.format(
            question=SAMPLE_QUESTION,
            characteristic_answer=SAMPLE_CHAR_ANSWER,
            agent_think=SAMPLE_THINK,
            agent_response=SAMPLE_RESPONSE,
        )
        assert len(prompt) > 100

        # Simulate a judge response and parse it
        simulated_response = VALID_JUDGE_OUTPUT
        result = json.loads(simulated_response)
        assert isinstance(result["hallucinations"], bool)
        assert isinstance(result["hallucination_type"], list)
        assert isinstance(result["rationale"], str)
