#!/usr/bin/env python3
"""Run and validate the reported calculation on licensed external inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def execute(script: str, *arguments: str) -> None:
    print(f"\n==> {script}", flush=True)
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--randomizations",
        type=int,
        help="Override the configured number of circular shifts.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    arguments = [
        "--data-dir",
        str(data_dir),
        "--results-dir",
        str(results_dir),
    ]
    if args.randomizations is not None:
        arguments.extend(
            ["--randomizations", str(args.randomizations)]
        )
    execute("run_core.py", *arguments)
    execute(
        "validate_release.py",
        "--results-dir",
        str(results_dir),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
