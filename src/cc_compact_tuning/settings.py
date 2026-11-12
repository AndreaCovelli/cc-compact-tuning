"""Where the transcripts, the store and the status line's rate-limit log live."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from CC_TUNING_* environment variables, e.g. CC_TUNING_DATA_DIR."""

    model_config = SettingsConfigDict(env_prefix="CC_TUNING_")

    projects_dir: Path = Path("~/.claude/projects")
    data_dir: Path = Path("~/.local/share/cc-compact-tuning")
    rate_log: Path = Path("~/.local/state/claude-statusline/rate-limits.tsv")

    @field_validator("projects_dir", "data_dir", "rate_log")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def store(self) -> Path:
        return self.data_dir / "store.json.gz"
