"""Budget tracking for API spending.

Abstracts over storage backends to track cumulative token usage.
Supports in-memory (default, no persistence) and HF Datasets (persistent).
"""

import json
import os
import time
from datetime import date
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Abstract store
# ---------------------------------------------------------------------------

class BudgetStore:
    """Base class for budget storage backends."""

    def get(self) -> dict:
        """Return {'tokens': int, 'date': str, 'reset_hour': int}."""
        raise NotImplementedError

    def save(self, data: dict) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory store (works everywhere, resets on restart)
# ---------------------------------------------------------------------------

class MemoryBudgetStore(BudgetStore):
    def __init__(self):
        self._lock = Lock()
        self._data = {
            "tokens": 0,
            "date": str(date.today()),
            "reset_hour": 0,
        }

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def save(self, data: dict) -> None:
        with self._lock:
            self._data.update(data)


# ---------------------------------------------------------------------------
# HF Datasets store (persistent across restarts and deployments)
# ---------------------------------------------------------------------------

class HFBudgetStore(BudgetStore):
    """Store budget data in a Hugging Face Dataset repo as a JSON file.

    Requires:
        pip install huggingface_hub
        huggingface-cli login (or HF_TOKEN env var)
        A dataset repo created at repo_id (private recommended).
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.token = os.getenv("HF_TOKEN")
        self._lock = Lock()
        self._cache: dict | None = None  # cache to avoid HF API on every call

    def get(self) -> dict:
        with self._lock:
            if self._cache is not None:
                return dict(self._cache)
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename="budget.json",
                    repo_type="dataset",
                    token=self.token,
                )
                self._cache = json.load(open(path))
                return dict(self._cache)
            except Exception:
                default = {
                    "tokens": 0,
                    "date": str(date.today()),
                    "reset_hour": 0,
                }
                self._cache = default
                return dict(default)

    def save(self, data: dict) -> None:
        with self._lock:
            self._cache = dict(data)
            try:
                from huggingface_hub import upload_file
                tmp = "/tmp/rag_budget.json"
                with open(tmp, "w") as f:
                    json.dump(data, f)
                upload_file(
                    path_or_fileobj=tmp,
                    path_in_repo="budget.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
            except Exception:
                pass  # silent fail — in-memory cache still tracks


# ---------------------------------------------------------------------------
# Budget manager (token counter + daily reset)
# ---------------------------------------------------------------------------

class BudgetManager:
    def __init__(
        self,
        store: BudgetStore | None = None,
        daily_cap: int = 500_000,
        daily_reset_hour: int = 0,  # UTC
    ):
        self.store = store or MemoryBudgetStore()
        self.daily_cap = daily_cap
        self.daily_reset_hour = daily_reset_hour

    def _today(self) -> str:
        return str(date.today())

    def _current_hour_utc(self) -> int:
        return int(time.strftime("%H", time.gmtime()))

    def track(self, tokens: int) -> None:
        """Record *tokens* used, respecting daily reset."""
        data = self.store.get()

        today = self._today()
        hour = self._current_hour_utc()

        # Daily reset
        if data["date"] != today and hour >= self.daily_reset_hour:
            data["tokens"] = 0
            data["date"] = today
            data["reset_hour"] = self.daily_reset_hour

        data["tokens"] += tokens
        self.store.save(data)

    def remaining(self) -> int:
        return max(0, self.daily_cap - self.store.get()["tokens"])

    def used(self) -> int:
        return self.store.get()["tokens"]

    def exceeded(self) -> bool:
        return self.remaining() <= 0
