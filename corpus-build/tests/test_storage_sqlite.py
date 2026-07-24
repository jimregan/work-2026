from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from corpus_build.storage.sqlite import SQLiteBackend
from storage_conformance import StorageConformanceSuite


class TestSQLiteBackend(StorageConformanceSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> Iterator[SQLiteBackend]:
        backend = SQLiteBackend(tmp_path / "corpus.db")
        yield backend
        backend.close()
