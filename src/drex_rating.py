# -*- coding: utf-8 -*-
"""
Optional stock rating from Nace's Drex decision model (default off).

Enabled by NACE_API_KEY. Drex answers one score question over the canonical
decision scale and returns a probability per level; the LLM still writes the
report. Any failure returns None and the LLM rates the stock as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from api import __version__
from src.report_language import localize_operation_advice
from src.schemas.decision_scale import (
    CANONICAL_DECISION_SCALE,
    DecisionScaleBand,
    decision_band_for_score,
)

logger = logging.getLogger(__name__)

DREX_URL = "https://console.nace.ai/v1/systemone"
DEFAULT_DREX_MODEL = "drex-max-finance"
DREX_TIMEOUT_SECONDS = 20
# Nace attributes Drex traffic per integration from X-Nace-Client.
DREX_CLIENT = f"daily_stock_analysis/{__version__}"

# Drex reads score criteria from low to high; the scale is stored high to low.
_LEVELS = tuple(reversed(CANONICAL_DECISION_SCALE))

_QUESTION = {
    "type": "score",
    "instructions": (
        "Rate this stock now on the canonical score and action scale defined above, "
        "using only the data in this request."
    ),
    "criteria": [f"{b.label_zh} ({b.signal_key}): {b.description_zh}" for b in _LEVELS],
}


@dataclass(frozen=True)
class DrexRating:
    model: str
    score: int
    band: DecisionScaleBand
    probabilities: Dict[str, float]
    request_id: Optional[str]

    def advice(self, language: str) -> str:
        return localize_operation_advice(self.band.label_zh, language)

    def prompt_note(self, language: str) -> str:
        """Tell the LLM the rating so the report text agrees with it."""
        advice = self.advice(language)
        if language == "zh":
            return (
                f"\n\n## 评级（已确定）\n本次评级由 {self.model} 给出：{advice}"
                f"（sentiment_score={self.score}，decision_type={self.band.decision_type}）。"
                "请按此填写 `sentiment_score`、`operation_advice`、`decision_type`，"
                "报告其余内容与该评级保持一致。"
            )
        return (
            f"\n\n## Rating (already decided)\nThe rating was decided by {self.model}: {advice} "
            f"(sentiment_score={self.score}, decision_type={self.band.decision_type}). "
            "Use it for `sentiment_score`, `operation_advice` and `decision_type`, "
            "and keep the rest of the report consistent with it."
        )

    def record(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "score": self.score,
            "signal_key": self.band.signal_key,
            "probabilities": self.probabilities,
            "request_id": self.request_id,
        }


def rate_with_drex(state: str) -> Optional[DrexRating]:
    """Ask Drex for the rating; None when NACE_API_KEY is unset or the request fails."""
    api_key = os.getenv("NACE_API_KEY", "").strip()
    if not api_key:
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
            json={"model": model, "state": state, "questions": {"rating": _QUESTION}},
            timeout=DREX_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["answers"]["rating"]
        level = float(answer["score"])  # expected level, 0 (sell) to 4 (strong buy)
        if not 0 <= level <= len(_LEVELS) - 1:
            raise ValueError(f"score out of range: {level}")
        probabilities = {
            band.signal_key: float(answer["probabilities"][str(index)])
            for index, band in enumerate(_LEVELS)
        }
    except Exception as exc:  # any failure falls back to the LLM rating
        logger.warning("[Drex] 评级请求失败，改由 LLM 评级: %s", exc)
        return None
    # Each level maps to the middle of its 20-point band, so band edges fall on half levels.
    score = int(10 + 20 * level)
    return DrexRating(model, score, decision_band_for_score(score), probabilities, payload.get("request_id"))
