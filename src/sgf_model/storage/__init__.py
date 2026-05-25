"""Snapshot persistence: save full ranking runs + configs as self-describing documents.

Backends:
    - FileStorage (default): one JSON file per snapshot under a base directory,
      plus an `index.json` for fast listing. No infrastructure required.
    - MongoStorage: pymongo-backed, one collection of snapshot documents.

Choose via the `SGF_STORAGE` environment variable:
    - unset or "file" → FileStorage at ~/.sgf-model/snapshots/
    - "file:/path"    → FileStorage at /path
    - "mongodb://..." → MongoStorage with that connection string

Or pass a Storage instance directly in code (preferred for tests).
"""

from sgf_model.storage.base import MODEL_VERSION, Snapshot, Storage
from sgf_model.storage.factory import default_storage, get_storage
from sgf_model.storage.file import FileStorage
from sgf_model.storage.mongo import MongoStorage

__all__ = [
    "MODEL_VERSION",
    "FileStorage",
    "MongoStorage",
    "Snapshot",
    "Storage",
    "default_storage",
    "get_storage",
]
