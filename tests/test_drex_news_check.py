# -*- coding: utf-8 -*-
"""Optional Drex news check in the legacy stock analysis (NACE_API_KEY)."""

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
from src.drex_news_check import check_news_with_drex


def _llm_reply(score, advice, decision_type):
    return json.dumps(
        {
            "sentiment_score": score,
            "trend_prediction": "看多",
            "operation_advice": advice,
            "decision_type": decision_type,
            "analysis_summary": "ok",
            "risk_warning": "估值偏高",
            "dashboard": {"core_conclusion": {"one_sentence": "ok"}},
        },
        ensure_ascii=False,
    )


def _drex_response(bearish, adverse):
    response = MagicMock()
    response.json.return_value = {
        "answers": {
            "tone": {"type": "choice", "choice": "neutral",
                     "probabilities": {"bearish": bearish, "neutral": 1 - bearish, "bullish": 0.0}},
            "adverse": {"type": "noul", "noul": adverse},
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


def _analyze(reply=None, news="茅台年报：营收首次下滑"):
    reply = reply or _llm_reply(72, "买入", "buy")
    analyzer = _analyzer()
    with patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"), \
         patch.object(analyzer, "_build_market_snapshot", return_value={}), \
         patch.object(analyzer, "_call_litellm", return_value=(reply, "gemini/gemini-3-flash-preview", {})):
        return analyzer.analyze({"code": "600519", "stock_name": "贵州茅台"}, news)


class CheckNewsWithDrexTestCase(unittest.TestCase):
    def test_without_key_or_news_sends_nothing(self):
        for key, news in (("", "news"), ("test-key", ""), ("test-key", None)):
            with patch.dict(os.environ, {"NACE_API_KEY": key}), patch("src.drex_news_check.requests.post") as post:
                self.assertIsNone(check_news_with_drex("贵州茅台", "600519", news))
            post.assert_not_called()

    def test_asks_tone_and_adverse_event_about_the_news(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key", "NACE_DEFAULT_DECISION_MODEL": ""}), \
             patch("src.drex_news_check.requests.post", return_value=_drex_response(0.2, 0.9)) as post:
            check = check_news_with_drex("贵州茅台", "600519", "营收首次下滑")

        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "drex-max-finance")
        self.assertIn("贵州茅台(600519)", body["state"])
        self.assertIn("营收首次下滑", body["state"])
        self.assertEqual(list(body["questions"]["tone"]["criteria"]), ["bearish", "neutral", "bullish"])
        self.assertEqual(body["questions"]["adverse"]["type"], "noul")
        self.assertEqual(post.call_args.kwargs["headers"], {
            "Authorization": "Bearer test-key",
            "X-Nace-Client": f"daily_stock_analysis/{__version__}",
            "User-Agent": f"daily_stock_analysis/{__version__}",
        })
        self.assertEqual((check.adverse_probability, check.tone, check.request_id), (0.9, "neutral", "req_test"))

    def test_only_a_confident_answer_blocks_a_buy(self):
        cases = [((0.7, 0.0), True), ((0.0, 0.7), True), ((0.69, 0.69), False)]
        for (bearish, adverse), blocks in cases:
            with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
                 patch("src.drex_news_check.requests.post", return_value=_drex_response(bearish, adverse)):
                self.assertEqual(check_news_with_drex("贵州茅台", "600519", "news").blocks_buy, blocks)

    def test_failures_return_none(self):
        http_error = MagicMock()
        http_error.raise_for_status.side_effect = requests.HTTPError("401 Client Error")
        failures = [
            {"side_effect": requests.Timeout("timed out")},
            {"return_value": http_error},
            {"return_value": MagicMock(json=MagicMock(return_value={"error": {}}))},
            {"return_value": _drex_response(0.2, 1.5)},
        ]
        for kwargs in failures:
            with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
                 patch("src.drex_news_check.requests.post", **kwargs):
                self.assertIsNone(check_news_with_drex("贵州茅台", "600519", "news"))


class AnalyzeWithDrexNewsCheckTestCase(unittest.TestCase):
    def test_no_key_keeps_llm_result(self):
        with patch.dict(os.environ, {"NACE_API_KEY": ""}), patch("src.drex_news_check.requests.post") as post:
            result = _analyze()

        post.assert_not_called()
        self.assertEqual((result.sentiment_score, result.operation_advice, result.decision_type), (72, "买入", "buy"))
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview")
        self.assertNotIn("drex_news_check", result.dashboard)

    def test_confident_bad_news_downgrades_buy_to_watch(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
             patch("src.drex_news_check.requests.post", return_value=_drex_response(0.4, 0.9)):
            result = _analyze()

        self.assertEqual((result.sentiment_score, result.operation_advice, result.decision_type), (59, "观望", "hold"))
        self.assertEqual(result.action, "watch")
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview, drex-max-finance")
        self.assertTrue(result.dashboard["drex_news_check"]["blocks_buy"])
        reason = result.dashboard["decision_stability"]["reason"]
        self.assertIn("drex-max-finance 新闻核查", reason)
        self.assertIn(reason, result.risk_warning)
        self.assertEqual(result.dashboard["decision_score_calibration"]["guardrail_reason"], reason)

    def test_unsure_or_non_buy_keeps_llm_rating_and_records_check(self):
        cases = [(_llm_reply(72, "买入", "buy"), (0.6, 0.6), (72, "买入", "buy")),
                 (_llm_reply(30, "减仓", "sell"), (0.9, 0.9), (30, "减仓", "sell"))]
        for reply, (bearish, adverse), expected in cases:
            with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
                 patch("src.drex_news_check.requests.post", return_value=_drex_response(bearish, adverse)):
                result = _analyze(reply)
            self.assertEqual((result.sentiment_score, result.operation_advice, result.decision_type), expected)
            self.assertEqual(result.dashboard["drex_news_check"]["adverse_probability"], adverse)
            self.assertNotIn("decision_stability", result.dashboard)

    def test_drex_failure_keeps_llm_result(self):
        with patch.dict(os.environ, {"NACE_API_KEY": "test-key"}), \
             patch("src.drex_news_check.requests.post", side_effect=requests.ConnectionError("down")):
            result = _analyze()

        self.assertTrue(result.success)
        self.assertEqual((result.sentiment_score, result.operation_advice, result.decision_type), (72, "买入", "buy"))
        self.assertEqual(result.model_used, "gemini/gemini-3-flash-preview")
        self.assertNotIn("drex_news_check", result.dashboard)


if __name__ == "__main__":
    unittest.main()
