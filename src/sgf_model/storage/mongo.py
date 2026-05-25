"""MongoDB snapshot storage."""

from __future__ import annotations

from typing import Any

from pymongo import DESCENDING, MongoClient

from sgf_model.storage.base import Snapshot

DEFAULT_DB: str = "sgf_model"
DEFAULT_COLLECTION: str = "snapshots"


class MongoStorage:
    """Snapshots as documents in a MongoDB collection.

    Each document is keyed by `snapshot_id`. The `created_at` field is indexed
    for efficient time-ordered listing. The collection schema is the same dict
    that `Snapshot.to_dict()` produces — no separate ORM layer.
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        db_name: str = DEFAULT_DB,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self._client: MongoClient = MongoClient(connection_string)
        self._collection = self._client[db_name][collection_name]
        # Idempotent; safe to call repeatedly.
        self._collection.create_index("snapshot_id", unique=True)
        self._collection.create_index([("created_at", DESCENDING)])

    def save(self, snapshot: Snapshot) -> str:
        self._collection.replace_one(
            {"snapshot_id": snapshot.snapshot_id},
            snapshot.to_dict(),
            upsert=True,
        )
        return snapshot.snapshot_id

    def get(self, snapshot_id: str) -> Snapshot:
        doc = self._collection.find_one({"snapshot_id": snapshot_id})
        if doc is None:
            raise KeyError(f"snapshot not found: {snapshot_id}")
        doc.pop("_id", None)  # strip Mongo's internal id
        return Snapshot.from_dict(doc)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = (
            self._collection.find(
                {},
                projection={
                    "_id": 0,
                    "snapshot_id": 1,
                    "created_at": 1,
                    "as_of_season": 1,
                    "league_config.name": 1,
                    "scoring_config.name": 1,
                    "rankings": 1,
                    "notes": 1,
                    "model_version": 1,
                },
            )
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        out: list[dict[str, Any]] = []
        for doc in cursor:
            out.append(
                {
                    "snapshot_id": doc["snapshot_id"],
                    "created_at": doc["created_at"],
                    "as_of_season": doc["as_of_season"],
                    "league_name": doc.get("league_config", {}).get("name", "custom"),
                    "scoring_name": doc.get("scoring_config", {}).get("name", "custom"),
                    "n_players": len(doc.get("rankings", [])),
                    "notes": doc.get("notes", ""),
                    "model_version": doc.get("model_version", "?"),
                }
            )
        return out

    def delete(self, snapshot_id: str) -> None:
        self._collection.delete_one({"snapshot_id": snapshot_id})
