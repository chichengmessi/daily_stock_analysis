# -*- coding: utf-8 -*-
"""Optional Drex rating in the legacy stock analysis (NACE_API_KEY)."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from api import __version__

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from src.analyzer import GeminiAnalyzer
from src.drex_rating import rate_with_drex

LLM_REPLY = json.dumps(
    {
        "sentiment_score": 72,
        "trend_prediction": "看多",
        "operation_advice": "买入",
        "decision_type": "buy",
        "analysis_summary": "ok",
        "dashboard": {"core_conclusion": {"one_sentence": "ok"}},
    },
    ensure_ascii=False,
)


def _drex_response(score, probabilities=None):
    response = MagicMock()
    response.json.return_value = {
        "answers": {
            "rating": {
                "type": "score",
                "score": score,
                "probabilities": probabilities
                or {"0": 0.6, "1": 0.3, "2": 0.1, "3": 0.0, "4": 0.0},
            }
        },
        "request_id": "req_test",
    }
    return response


def _analyzer():
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._config_override = SimpleNamespace(
        generation_backend="litellm",
        generation_fallback_backend="litellm",
        litellm_model="gemini/gemini-3-flash-preview",
        litellm_fallback_models=[],
        llm_model_list=[],
        report_language="zh",
        gemini_request_delay=0,
        llm_temperature=0.7,
        report_integrity_enabled=False,
        report_integrity_retry=0,
    )
    return analyzer


def _analyze(analyzer):
    prompts = []

    def _llm(prompt, generation_config, **kwargs):
        prompts.append(prompt)
        return LLM_REPLY, "gemini/gemini-3-flash-preview", {}

    with patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"), \
         patch.object(analyzer, "_build_market_snapshot", return_value={}), \
         patch.object(analyzer, "_call_litellm", side_effect=_llm):
        result = analyzer.analyze({"code": "600519", "stock_name": "贵州茅台"})
    return result, prompts[0]


class RateWithDrexTestCase(unittest.TestCase):
    def test_without_key_sends_nothing(self):
        with patch.dict(os.environ, {"NACE_API_KEY": ""}), patch("src.drex_rating.requests.post") as post:
            self.assertIsNone(rate_with_drex("state"))
        post.assert_not_called()

    def test_score_question_runs_from_sell_to_strong_buy(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key", "NACE_DEFAULT_DECISION_MODEL": ""}), \
             patch("src.drex_rating.requests.post", return_value=_drex_response(0.5)) as post:
            rating = rate_with_drex("state")

        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "drex-max-finance")
        self.assertEqual(body["state"], "state")
        criteria = body["questions"]["rating"]["criteria"]
        self.assertTrue(criteria[0].startswith("卖出"))
        self.assertTrue(criteria[-1].startswith("强烈买入"))
        self.assertEqual(post.call_args.kwargs["headers"], {
            "Authorization": "Bearer test-key",
            "X-Nace-Client": f"daily_stock_analysis/{__version__}",
            "User-Agent": f"daily_stock_analysis/{__version__}",
        })
        self.assertEqual(rating.probabilities["sell"], 0.6)
        self.assertEqual(rating.request_id, "req_test")

    def test_expected_level_maps_to_canonical_band(self):
        cases = [(0, 10, "sell"), (0.49, 19, "sell"), (0.5, 20, "reduce"), (2.49, 59, "watch"),
                 (2.5, 60, "buy"), (4, 90, "strong_buy")]
        for level, score, signal in cases:
            with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
                 patch("src.drex_rating.requests.post", return_value=_drex_response(level)):
                rating = rate_with_drex("state")
            self.assertEqual((rating.score, rating.band.signal_key), (score, signal), level)

    def test_failures_return_none(self):
        http_error = MagicMock()
        http_error.raise_for_status.side_effect = requests.HTTPError("401 Client Error")
        failures = [
            {"side_effect": requests.Timeout("timed out")},
            {"return_value": http_error},
            {"return_value": MagicMock(json=MagicMock(return_value={"error": {}}))},
            {"return_value": _drex_response(7)},
        ]
        for kwargs in failures:
            with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
                 patch("src.drex_rating.requests.post", **kwargs):
                self.assertIsNone(rate_with_drex("state"))


class AnalyzeWithDrexTestCase(unittest.TestCase):
    def test_no_key_keeps_llm_rating_and_prompt(self):
        with patch.dict(os.environ, {"NACE_API_KEY": ""}), patch("src.drex_rating.requests.post") as post:
            result, prompt = _analyze(_analyzer())

        post.assert_not_called()
        self.assertEqual(prompt, "prompt")
        self.assertEqual((result.sentiment_score, result.operation_advice), (72, "买入"))
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview")
        self.assertNotIn("drex_rating", result.dashboard)

    def test_drex_rating_replaces_llm_rating(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
             patch("src.drex_rating.requests.post", return_value=_drex_response(1.2)) as post:
            result, prompt = _analyze(_analyzer())

        self.assertEqual(post.call_args.kwargs["json"]["state"], "system\n\nprompt")
        self.assertIn("本次评级由 drex-max-finance 给出：减仓", prompt)
        self.assertEqual(result.sentiment_score, 34)
        self.assertEqual(result.operation_advice, "减仓")
        self.assertEqual((result.decision_type, result.action), ("sell", "reduce"))
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview, drex-max-finance")
        self.assertEqual(result.dashboard["drex_rating"]["signal_key"], "reduce")
        self.assertEqual(result.dashboard["core_conclusion"], {"one_sentence": "ok"})

    def test_drex_failure_keeps_llm_rating_and_prompt(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
             patch("src.drex_rating.requests.post", side_effect=requests.ConnectionError("down")):
            result, prompt = _analyze(_analyzer())

        self.assertEqual(prompt, "prompt")
        self.assertTrue(result.success)
        self.assertEqual((result.sentiment_score, result.operation_advice), (72, "买入"))
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview")


if __name__ == "__main__":
    unittest.main()
