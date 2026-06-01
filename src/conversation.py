"""Conversation logging for the UI GreenMetric RAG system.

Logs user queries and generated answers to a HF Dataset for
analysis and monitoring. Uses the same repo as budget tracking.
"""

import json
import os
import time
from datetime import datetime, timezone
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# HF logger
# ---------------------------------------------------------------------------

class ConversationLogger:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.token = os.getenv("HF_TOKEN")
        self._buffer: list[dict] = []
        self._lock = Lock()

    def log(self, entry: dict) -> None:
        with self._lock:
            self._buffer.append(entry)

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            try:
                from huggingface_hub import upload_file, hf_hub_download

                tmp = "/tmp/rag_conversations.jsonl"

                try:
                    existing = hf_hub_download(
                        repo_id=self.repo_id,
                        filename="conversations.jsonl",
                        repo_type="dataset",
                        token=self.token,
                    )
                    import shutil
                    shutil.copy(existing, tmp)
                except Exception:
                    open(tmp, "w").close()  # fresh file

                with open(tmp, "a") as f:
                    for e in self._buffer:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")

                upload_file(
                    path_or_fileobj=tmp,
                    path_in_repo="conversations.jsonl",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                self._buffer.clear()
            except Exception:
                pass  # silent — conversation logging is best-effort


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_logger: ConversationLogger | None = None


def get_logger() -> ConversationLogger | None:
    global _logger
    if _logger is None:
        repo = os.getenv("RAG_BUDGET_REPO", "")
        if repo and os.getenv("HF_TOKEN"):
            _logger = ConversationLogger(repo)
    return _logger


def log_conversation(
    query: str,
    answer: str,
    route: dict,
    chunks: list[str],
    tokens: int,
) -> None:
    logger = get_logger()
    if logger is None:
        return
    logger.log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_prompt": query,
        "response": answer,
        "router_source": route.get("source", "?"),
        "router_csv_source": route.get("csv_source"),
        "query_type": route.get("query_type", "?"),
        "token_usage": tokens,
        "chunks": chunks,
    })
