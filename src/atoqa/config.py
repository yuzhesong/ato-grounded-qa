"""Runtime configuration.

Everything that changes behaviour is settable by environment variable so that
the same code path runs in tests, CI, and against a live provider.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATOQA_", env_file=".env", extra="ignore")

    # offline | openai | azure
    provider: str = "offline"

    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    chat_deployment: str = "gpt-4o-mini"
    embed_deployment: str = "text-embedding-3-small"

    # Retrieval
    top_k: int = 6
    dense_weight: float = 0.5
    rrf_k: int = 60

    # Control thresholds. These are the knobs that decide whether the system
    # answers at all; they are deliberately explicit rather than buried.
    min_retrieval_score: float = 0.18
    min_groundedness: float = 0.60
    # Bar for treating an apparently out-of-scope question as tax after all.
    # Deliberately above min_retrieval_score: a marginal match is not evidence
    # that the corpus covers the topic.
    min_rescue_score: float = 0.30

    # Tax facts are year-scoped. The system never silently guesses a year: if the
    # question does not name one, this default is used AND surfaced in the answer.
    default_income_year: str = "2025-26"

    raw_dir: Path = REPO_ROOT / "data" / "raw"
    index_dir: Path = REPO_ROOT / "data" / "index"
    trace_dir: Path = REPO_ROOT / "data" / "traces"

    user_agent: str = "atoqa-portfolio-demo/0.1 (+https://github.com/yuzhesong)"


settings = Settings()
