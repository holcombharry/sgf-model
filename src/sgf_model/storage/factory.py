"""Choose a storage backend from env var or URL."""

from __future__ import annotations

import os

from sgf_model.storage.base import Storage
from sgf_model.storage.file import FileStorage
from sgf_model.storage.mongo import MongoStorage


def get_storage(spec: str | None = None) -> Storage:
    """Resolve a Storage instance from a spec string.

    Accepted forms:
        None or "" or "file"      → FileStorage at default location
        "file:/some/path"          → FileStorage at /some/path
        "mongodb://host:port/..."  → MongoStorage with that connection string

    For mongo, the database name defaults to `sgf_model` and the collection
    to `snapshots`. Override by constructing `MongoStorage` directly.
    """
    if not spec or spec == "file":
        return FileStorage()
    if spec.startswith("file:"):
        return FileStorage(base_dir=spec[len("file:") :])
    if spec.startswith("mongodb://") or spec.startswith("mongodb+srv://"):
        return MongoStorage(connection_string=spec)
    raise ValueError(f"Unrecognized storage spec: {spec!r}")


def default_storage() -> Storage:
    """Pick storage from the `SGF_STORAGE` env var, falling back to FileStorage."""
    return get_storage(os.environ.get("SGF_STORAGE"))
