"""Canonical records, atomic writes and same-process workflow integrity checks.

Hashes detect changes, not authenticity. Only use bundles from trusted authors.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("INVALID_RELATIVE_PATH")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("PATH_ESCAPES_ROOT")
    return result


def seal(root: Path, files: list[str]) -> dict[str, str]:
    hashes = {f: sha_file(safe_child(root, f)) for f in sorted(files)}
    write_json(root / "manifest.json", {"schema": "phm.files.v1", "files": hashes})
    return hashes


def verify(root: Path) -> dict[str, str]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "phm.files.v1" or not manifest.get("files"):
        raise ValueError("INVALID_MANIFEST")
    for name, expected in manifest["files"].items():
        file = safe_child(root, name)
        if not file.is_file() or sha_file(file) != expected:
            raise ValueError(f"INTEGRITY_FAILURE: {name}")
    return manifest["files"]


@contextlib.contextmanager
def run_lock(root: Path) -> Iterator[None]:
    """Linux advisory lock, released by the OS if the owning process exits."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("RUN_ALREADY_LOCKED") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
