# -*- coding: utf-8 -*-
"""
Optional news check from Nace's Drex decision model (default off).

Enabled by NACE_API_KEY. Drex reads the stock's news search results and answers two
typed questions, each with a probability: the overall tone of the news and whether it
reports a material negative event. The LLM still rates the stock and writes the report;
a buy call is downgraded to watch only when Drex is confident the news is bad. Without
news or on any failure the result is None and the analysis is unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from api import __version__

logger = logging.getLogger(__name__)

DREX_URL = "https://console.nace.ai/v1/systemone"
DEFAULT_DREX_MODEL = "drex-max-finance"
DREX_TIMEOUT_SECONDS = 20
# Nace attributes Drex traffic per integration from X-Nace-Client.
DREX_CLIENT = f"daily_stock_analysis/{__version__}"
# Only a confident answer changes the rating; below it the LLM's call stands.
GATE_PROBABILITY = 0.7

QUESTIONS = {
    "tone": {
        "type": "choice",
        "instructions": (
            "Judging only from these news search results, what do they imply for this "
            "stock's share price over the next two weeks?"
        ),
        "criteria": {
            "bearish": (
                "Material negative company news dominates: earnings or guidance miss, lawsuit, "
                "regulatory action, analyst downgrade, large shareholder selling"
            ),
            "neutral": "No material company news, mixed news, or routine coverage only",
            "bullish": (
                "Material positive company news dominates: earnings or guidance beat, major "
                "contract or approval, analyst upgrade, buyback"
            ),
        },
    },
    "adverse": {
        "type": "noul",
        "instructions": (
            "Do these news search results report a specific, material negative event for this "
            "company, such as an earnings or revenue miss, a guidance cut, a lawsuit, a "
            "regulatory penalty or investigation, an accounting problem, a major product "
            "failure, an analyst downgrade, or large shareholder selling (减持)?"
        ),
    },
}


def news_state(name: str, code: str, news_context: str) -> str:
    return f"{name}({code}) 新闻搜索结果 / news search results:\n\n{news_context}"


@dataclass(frozen=True)
class DrexNewsCheck:
    model: str
    tone_probabilities: Dict[str, float]
    adverse_probability: float
    request_id: Optional[str]

    @property
    def tone(self) -> str:
        return max(self.tone_probabilities, key=self.tone_probabilities.get)

    @property
    def blocks_buy(self) -> bool:
        return max(self.tone_probabilities["bearish"], self.adverse_probability) >= GATE_PROBABILITY

    def reason(self, language: str) -> str:
        bearish, adverse = self.tone_probabilities["bearish"], self.adverse_probability
        if language == "zh":
            return (
                f"{self.model} 新闻核查：新闻偏利空概率 {bearish:.2f}，"
                f"重大负面事件概率 {adverse:.2f}，买入结论先按观察处理。"
            )
        return (
            f"{self.model} news check: bearish news probability {bearish:.2f}, "
            f"material negative event probability {adverse:.2f}; treat the buy call as watch-only."
        )

    def record(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "tone": self.tone,
            "tone_probabilities": self.tone_probabilities,
            "adverse_probability": self.adverse_probability,
            "blocks_buy": self.blocks_buy,
            "request_id": self.request_id,
        }


def check_news_with_drex(name: str, code: str, news_context: Optional[str]) -> Optional[DrexNewsCheck]:
    """Ask Drex about the news; None without NACE_API_KEY or news, or when the request fails."""
    api_key = os.getenv("NACE_API_KEY", "").strip()
    if not api_key or not (news_context or "").strip():
        return None
    model = os.getenv("NACE_DEFAULT_DECISION_MODEL", "").strip() or DEFAULT_DREX_MODEL
    try:
        response = requests.post(
            DREX_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Nace-Client": DREX_CLIENT,
                "User-Agent": DREX_CLIENT,
            },
            json={"model": model, "state": news_state(name, code, news_context), "questions": QUESTIONS},
            timeout=DREX_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answers = payload["answers"]
        tone = {key: float(answers["tone"]["probabilities"][key]) for key in QUESTIONS["tone"]["criteria"]}
        adverse = float(answers["adverse"]["noul"])
        if not all(0 <= p <= 1 for p in (*tone.values(), adverse)):
            raise ValueError(f"probability out of range: {tone}, {adverse}")
    except Exception as exc:  # any failure leaves the LLM's analysis unchanged
        logger.warning("[Drex] 新闻核查请求失败，保持 LLM 结论: %s", exc)
        return None
    return DrexNewsCheck(model, tone, adverse, payload.get("request_id"))
