"""
Token budget monitor — adapted from OMPAminnow's TokenBudget.
Soft monitoring only: never blocks retrieval, fires WARNING/TREND/CRITICAL alerts.
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

WARN_MULTIPLIER = 2.0
TREND_WINDOW = 5
CRITICAL_TOKENS = 4000


class BudgetStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    TREND = "trend"
    CRITICAL = "critical"


@dataclass
class BudgetCheck:
    status: BudgetStatus
    tokens: int
    message: str


class ContextBudget:
    def __init__(self, critical_threshold: int = CRITICAL_TOKENS):
        self.critical_threshold = critical_threshold
        self._history: deque[int] = deque(maxlen=20)
        self._consecutive_growth = 0
        self._last_tokens = 0

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def record(self, tokens: int) -> BudgetCheck:
        self._history.append(tokens)

        if tokens >= self.critical_threshold:
            logger.warning("CRITICAL token usage: %d", tokens)
            self._consecutive_growth = 0
            self._last_tokens = tokens
            return BudgetCheck(
                BudgetStatus.CRITICAL,
                tokens,
                f"Critical: {tokens} tokens exceeds threshold {self.critical_threshold}",
            )

        if self._last_tokens > 0 and tokens > self._last_tokens:
            self._consecutive_growth += 1
        else:
            self._consecutive_growth = 0
        self._last_tokens = tokens

        if self._consecutive_growth >= TREND_WINDOW:
            logger.warning("TREND: token usage growing for %d consecutive rounds", self._consecutive_growth)
            return BudgetCheck(
                BudgetStatus.TREND,
                tokens,
                f"Trend: usage growing {self._consecutive_growth} rounds",
            )

        if len(self._history) >= 5:
            mean = sum(self._history) / len(self._history)
            if mean > 0 and tokens > mean * WARN_MULTIPLIER:
                return BudgetCheck(
                    BudgetStatus.WARNING,
                    tokens,
                    f"Warning: {tokens} tokens is {tokens / mean:.1f}x above mean",
                )

        return BudgetCheck(BudgetStatus.OK, tokens, "ok")

    def stats(self) -> dict:
        mean = sum(self._history) / len(self._history) if self._history else 0
        return {
            "mean_tokens": round(mean, 1),
            "last_tokens": self._last_tokens,
            "consecutive_growth": self._consecutive_growth,
        }
