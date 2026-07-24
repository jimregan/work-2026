from __future__ import annotations

from pathlib import Path

import pytest

from corpus_build.storage.filesystem import FilesystemBackend
from storage_conformance import StorageConformanceSuite


class TestFilesystemBackend(StorageConformanceSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> FilesystemBackend:
        return FilesystemBackend(tmp_path)
