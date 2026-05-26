"""Snapshot data model + storage protocol.

A snapshot captures a *full ranking run*: the configs that produced it, the
data window, the resulting rankings. Two design goals:

1. **Self-describing.** Anyone reading a snapshot can reconstruct exactly
   what produced it without checking other files. Includes a model version
   string so old snapshots aren't misinterpreted after model changes.

2. **Transport-portable.** JSON-serializable in / out. Backends differ only
   in how the documents land on disk / in a database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

# Bump on breaking changes to the pipeline (curve method, projection scheme,
# scoring schema, etc.). Old snapshots remain readable; consumers can filter
# by version if needed.
MODEL_VERSION: str = "0.1.0"


@dataclass
class Snapshot:
    """A single ranking run: configs + outputs, fully self-describing."""

    as_of_season: int
    n_future_seasons: int
    league_config: dict[str, Any]
    scoring_config: dict[str, Any]
    discount_rate: float
    team_overrides: dict[str, str]
    rankings: list[dict[str, Any]]
    notes: str = ""
    snapshot_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Snapshot:
        # Tolerate missing optional fields when reading old snapshots.
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def summary(self) -> dict[str, Any]:
        """Lightweight metadata used by `list()` — no rankings payload."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "as_of_season": self.as_of_season,
            "league_name": self.league_config.get("name", "custom"),
            "scoring_name": self.scoring_config.get("name", "custom"),
            "n_players": len(self.rankings),
            "notes": self.notes,
            "model_version": self.model_version,
        }


class Storage(Protocol):
    """Common interface for snapshot persistence backends."""

    def save(self, snapshot: Snapshot) -> str: ...

    def get(self, snapshot_id: str) -> Snapshot: ...

    def list(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def delete(self, snapshot_id: str) -> None: ...
