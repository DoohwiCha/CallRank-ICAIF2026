#!/usr/bin/env python3
"""Refresh byte counts and SHA-256 hashes in the aggregate manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "aggregate_results"
MANIFEST = AGGREGATE / "aggregate_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {
        str(path.relative_to(AGGREGATE))
        for path in AGGREGATE.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and not any(part.startswith(".") for part in path.relative_to(AGGREGATE).parts)
    }
    files: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        path = AGGREGATE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest["files"] = files
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {MANIFEST} with {len(files)} file hashes")


if __name__ == "__main__":
    main()
