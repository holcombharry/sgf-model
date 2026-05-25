"""Filesystem snapshot storage: one JSON per snapshot under a base directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sgf_model.storage.base import Snapshot


class FileStorage:
    """Snapshots as JSON files. Default base: ~/.sgf-model/snapshots/.

    Each snapshot lives at `<base>/<snapshot_id>.json`. The `list()` operation
    scans the directory and reads each file's summary — fast enough for the
    expected scale (dozens to hundreds of snapshots).
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        base = Path(base_dir) if base_dir else Path.home() / ".sgf-model" / "snapshots"
        base.mkdir(parents=True, exist_ok=True)
        self.base_dir = base

    def _path(self, snapshot_id: str) -> Path:
        return self.base_dir / f"{snapshot_id}.json"

    def save(self, snapshot: Snapshot) -> str:
        path = self._path(snapshot.snapshot_id)
        path.write_text(json.dumps(snapshot.to_dict(), default=str))
        return snapshot.snapshot_id

    def get(self, snapshot_id: str) -> Snapshot:
        path = self._path(snapshot_id)
        if not path.exists():
            raise KeyError(f"snapshot not found: {snapshot_id}")
        return Snapshot.from_dict(json.loads(path.read_text()))

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        files = sorted(
            self.base_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        out: list[dict[str, Any]] = []
        for f in files:
            try:
                snap = Snapshot.from_dict(json.loads(f.read_text()))
                out.append(snap.summary())
            except Exception:
                # Skip unreadable / malformed snapshots rather than crashing list.
                continue
        return out

    def delete(self, snapshot_id: str) -> None:
        path = self._path(snapshot_id)
        if path.exists():
            path.unlink()
