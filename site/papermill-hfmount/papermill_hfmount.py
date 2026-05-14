"""Papermill I/O handler that mounts HuggingFace repos via hf-mount."""

import os
import subprocess
from pathlib import Path

MOUNT_BASE = Path.home() / ".cache" / "papermill-hfmount"


def _parse_hf_path(path):
    """Split 'hf://owner/repo/a/b.ipynb' into ('owner/repo', 'a/b.ipynb')."""
    without_scheme = path[len("hf://"):]
    parts = without_scheme.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"hf:// path must be at least hf://owner/repo, got: {path!r}")
    repo_id = f"{parts[0]}/{parts[1]}"
    file_path = parts[2] if len(parts) > 2 else ""
    return repo_id, file_path


def _mount_point(repo_id):
    owner, repo = repo_id.split("/", 1)
    return MOUNT_BASE / owner / repo


def _ensure_mounted(repo_id):
    mount_point = _mount_point(repo_id)
    mount_point.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["hf-mount", "start", "repo", repo_id, str(mount_point)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.lower()
        stdout = result.stdout.lower()
        # Treat "already mounted" as success
        if "already" not in stderr and "already" not in stdout:
            raise RuntimeError(
                f"hf-mount failed for {repo_id!r}:\n{result.stderr or result.stdout}"
            )

    return mount_point


class HFMountHandler:
    @classmethod
    def read(cls, path):
        repo_id, file_path = _parse_hf_path(path)
        local = _ensure_mounted(repo_id) / file_path
        return local.read_text()

    @classmethod
    def write(cls, content, path):
        repo_id, file_path = _parse_hf_path(path)
        local = _ensure_mounted(repo_id) / file_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content)

    @classmethod
    def pretty_path(cls, path):
        return path

    @classmethod
    def listdir(cls, path):
        repo_id, file_path = _parse_hf_path(path)
        mount_point = _ensure_mounted(repo_id)
        local = mount_point / file_path if file_path else mount_point
        base = f"hf://{repo_id}"
        if file_path:
            base = f"{base}/{file_path}"
        return [f"{base}/{entry}" for entry in os.listdir(local)]
