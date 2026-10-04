#!/usr/bin/env python3
"""Validate the public CallRank artifact and optional fresh-run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_config, rank_summary, save_json
from portfolio import strategy_summary
from recompute_published_results import verify_aggregate
from callrank_features import RAW_DISPLACEMENT, UNIT_DIRECTION


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODE = load_config().get("release_mode") == "public_camera_ready"
MAX_FILE_BYTES = 8 * 1024 * 1024
TEXT_SUFFIXES = {
    "",
    ".gitignore",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yml",
    ".yaml",
    ".cff",
}
LOCAL_ONLY_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}
AGGREGATE_REQUIRED = {
    "README.md",
    "alpha_lag_sensitivity.csv",
    "baseline_portfolio_daily_returns.csv",
    "baseline_portfolio_summary.csv",
    "call_date_audit_summary.json",
    "canonical_data_summary.json",
    "cleanroom_integrity_summary.json",
    "circular_shift_tests.csv",
    "cost_sensitivity.csv",
    "direction_concentration_summary.json",
    "encoder_summary.csv",
    "factor_concentration_summary.json",
    "factor_daily_ff5_momentum.csv",
    "factor_regressions.csv",
    "financial_controls.csv",
    "held_out_sector_tests.csv",
    "history_reference_tests.csv",
    "identification_expansion_summary.json",
    "identity_control_tests.csv",
    "identity_control_monthly_ic.csv",
    "identity_control_shift_null.csv",
    "issuer_identity_probe.csv",
    "grouped_signal_month_exclusions.csv",
    "leave_one_sector_out.csv",
    "paired_ic_tests.csv",
    "pca_ridge_sensitivity.csv",
    "portfolio_comparison_tests.csv",
    "portfolio_daily_returns.csv",
    "portfolio_summary.csv",
    "portfolio_yearly.csv",
    "raw_cost_sensitivity.csv",
    "raw_portfolio_daily_returns.csv",
    "raw_portfolio_summary.csv",
    "random_sector_null.csv",
    "random_sector_null_summary.json",
    "raw_robustness.csv",
    "representation_results.csv",
    "sector_concentration_summary.csv",
    "static_sector_benchmark.csv",
    "structured_news_tests.csv",
    "top1_selection_frequency.csv",
    "training_correction_ablation.csv",
    "wrong_firm_placebo_draws.csv",
    "temporal_circular_shift_null.csv",
    "aggregate_summary.json",
    "figures/callrank_sector_allocation.pdf",
    "figures/callrank_sector_allocation.png",
    "figures/callrank_time_alignment.pdf",
    "figures/callrank_time_alignment.png",
    "figures/callrank_method.pdf",
    "figures/callrank_method.png",
    "figures/callrank_history_reference.pdf",
    "figures/callrank_history_reference.png",
    "figures/callrank_robustness.pdf",
    "figures/callrank_robustness.png",
}
PUBLIC_INPUT_FILES = {
    f"{form}_{words}.npz" for form in ("raw", "unit")
    for words in (128, 160, 200)
} | {
    "train_keys.csv", "eval_keys.csv", "sector_total_return_indices.csv",
    "daily_factors.csv", "DGS3MO.csv",
}
PUBLIC_SECTORS = {
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
}


def is_public_path(path: Path) -> bool:
    """Return whether a path belongs to the distributable repository tree."""
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        relative = path
    return not any(part in LOCAL_ONLY_PARTS for part in relative.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(
    checks: list[dict[str, Any]],
    label: str,
    passed: bool,
    detail: Any = "",
) -> None:
    checks.append(
        {"check": label, "passed": bool(passed), "detail": detail}
    )


def tracked_files() -> list[Path]:
    files = {
        path
        for path in ROOT.rglob("*")
        if path.is_file() and is_public_path(path)
    }
    if (ROOT / ".git").exists():
        files.update(
            ROOT / raw.decode()
            for raw in subprocess.check_output(
                ["git", "ls-files", "-z"], cwd=ROOT
            ).split(b"\0")
            if raw
        )
    return sorted(files)


def text_findings() -> list[str]:
    scanner = Path(__file__).resolve()
    patterns = {
        "absolute macOS path": re.compile(r"/Users/[^/\s]+/"),
        "absolute Windows path": re.compile(
            r"[A-Za-z]:\\\\Users\\\\[^\\\\\s]+"
        ),
        "email address": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
        ),
        "credential assignment": re.compile(
            r"(?:api[_-]?key|apikey|password|access[_-]?token)"
            r"['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}",
            re.I,
        ),
        "credential in URL": re.compile(
            r"[?&](?:apikey|api_key|token)=[^&\s]+", re.I
        ),
        "development tree": re.compile(
            r"(?:submissions/v\d|projects/icaif|\.gjc/)", re.I
        ),
        "obsolete primary portfolio": re.compile(
            r"(?:primary|headline|main)\s+(?:equal[- ]weight\s+)?"
            r"top[- ]?(?:three|3)|equal[- ]weight\s+top[- ]?(?:three|3)",
            re.I,
        ),
        "obsolete artifact version": re.compile(
            r"(?:frozen|CallRank|Aggregate)\s+v[0-9]+"
            r"|run_v[0-9]+|v[0-9]+_features|callrank_v[0-9]+",
            re.I,
        ),
        "removed two-sector endpoint": re.compile(
            r"\b(?:top|bottom)\s*2\b|\b(?:top2|bottom2)\b",
            re.I,
        ),
    }
    findings: list[str] = []
    for path in tracked_files():
        if (
            not path.is_file()
            or path.resolve() == scanner
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns.items():
            matches = list(pattern.finditer(content))
            if PUBLIC_MODE and label == "email address":
                permitted = set(public_release_metadata().get("maintainer_git_emails", []))
                matches = [match for match in matches if match.group(0) not in permitted]
            if matches:
                findings.append(
                    f"{label}: {path.relative_to(ROOT)}"
                )
    return findings


def public_release_metadata() -> dict[str, Any]:
    path = ROOT / "PUBLIC_RELEASE.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def public_input_checks() -> tuple[list[dict[str, Any]], set[Path]]:
    """Approve only the hash-bound sector sufficient-statistic replay inputs."""
    checks: list[dict[str, Any]] = []
    directory = ROOT / "public_inputs"
    manifest_path = directory / "manifest.json"
    add(checks, "public sufficient-statistic input manifest exists", manifest_path.is_file())
    if not manifest_path.is_file():
        return checks, set()
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    file_names = set(files) if isinstance(files, dict) else set()
    add(checks, "public replay input file allowlist is exact",
        isinstance(files, dict) and set(files) == PUBLIC_INPUT_FILES,
        sorted(file_names.symmetric_difference(PUBLIC_INPUT_FILES)))
    if not isinstance(files, dict):
        files = {}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path not in {manifest_path, directory / "README.md"}}
    add(checks, "public replay input directory has no unapproved files", actual == PUBLIC_INPUT_FILES,
        sorted(actual.symmetric_difference(PUBLIC_INPUT_FILES)))
    approved: set[Path] = set()
    for name in sorted(PUBLIC_INPUT_FILES):
        path = directory / name
        description = files.get(name, {})
        valid = path.is_file() and not path.is_symlink()
        valid = valid and sha256(path) == description.get("sha256")
        valid = valid and path.stat().st_size == description.get("bytes")
        add(checks, f"public replay input checksum: {name}", valid)
        if not valid:
            continue
        try:
            if name.endswith(".npz"):
                with np.load(path, allow_pickle=False) as arrays:
                    schema_valid = set(arrays.files) == {"train_gram", "eval_train_gram"}
                    train = arrays["train_gram"]
                    evaluate = arrays["eval_train_gram"]
                    schema_valid = (schema_valid and train.shape == (703, 703)
                                    and evaluate.shape == (265, 703)
                                    and train.dtype == np.dtype("float64")
                                    and evaluate.dtype == np.dtype("float64")
                                    and np.isfinite(train).all() and np.isfinite(evaluate).all()
                                    and np.allclose(train, train.T, rtol=1e-12, atol=1e-10))
            else:
                frame = pd.read_csv(path)
                if name in {"train_keys.csv", "eval_keys.csv"}:
                    expected_rows = 703 if name == "train_keys.csv" else 265
                    schema_valid = (len(frame) == expected_rows
                                    and set(frame) == {"sector", "month", "entry_date", "exit_date", "excess_return"}
                                    and set(frame.sector).issubset(PUBLIC_SECTORS)
                                    and not frame[["sector", "month"]].duplicated().any())
                    if name == "train_keys.csv":
                        schema_valid = schema_valid and pd.to_datetime(frame.exit_date).le("2020-12-01").all()
                    else:
                        schema_valid = schema_valid and pd.to_datetime(frame.month).ge("2023-11-01").all()
                elif name == "sector_total_return_indices.csv":
                    schema_valid = set(frame) == {"Date", "SPY", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"}
                elif name == "daily_factors.csv":
                    schema_valid = {name.lower() for name in frame} == {"date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"}
                else:
                    schema_valid = set(frame).issubset({"observation_date", "DATE", "DGS3MO"})
                schema_valid = schema_valid and not frame.empty
            add(checks, f"public replay input schema: {name}", schema_valid)
            if schema_valid:
                approved.add(path.resolve())
        except (ValueError, KeyError, OSError, AttributeError) as error:
            add(checks, f"public replay input schema: {name}", False, type(error).__name__)
    if any(not check["passed"] for check in checks):
        approved.clear()
    return checks, approved


def git_identity_findings() -> list[str]:
    if not (ROOT / ".git").exists():
        return []
    output = subprocess.check_output(
        [
            "git",
            "log",
            "HEAD",
            "--format=%an%x09%ae%x09%cn%x09%ce",
        ],
        cwd=ROOT,
        text=True,
    )
    if PUBLIC_MODE:
        metadata = public_release_metadata()
        names = set(metadata.get("maintainer_git_names", []))
        emails = set(metadata.get("maintainer_git_emails", []))
        return [line for line in output.splitlines()
                if line and not (len(line.split("\t")) == 4
                                 and line.split("\t")[0] in names
                                 and line.split("\t")[1] in emails
                                 and line.split("\t")[2] in names
                                 and line.split("\t")[3] in emails)]
    allowed = (
        "Anonymous Authors\tanonymous@invalid.example\t"
        "Anonymous Authors\tanonymous@invalid.example"
    )
    return [
        line for line in output.splitlines() if line and line != allowed
    ]


def git_history_shape() -> dict[str, Any]:
    """Describe the intentionally history-clean anonymous review repository."""
    if not (ROOT / ".git").exists():
        return {
            "applicable": False,
            "passed": True,
            "detail": "source archive without Git metadata",
        }
    commits = int(
        subprocess.check_output(
            ["git", "rev-list", "--count", "--all"], cwd=ROOT, text=True
        ).strip()
    )
    roots = subprocess.check_output(
        ["git", "rev-list", "--max-parents=0", "--all"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    branches = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    tags = subprocess.check_output(
        ["git", "tag", "--list"], cwd=ROOT, text=True
    ).splitlines()
    head_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    detail = {
        "commits": commits,
        "roots": len(roots),
        "branches": branches,
        "tags": tags,
        "head_branch": head_branch,
    }
    return {
        "applicable": True,
        "passed": (commits >= 1 if PUBLIC_MODE else commits == 1)
        and len(roots) == 1
        and branches == ["main"]
        and (PUBLIC_MODE or not tags)
        and head_branch == "main",
        "detail": detail,
    }


def binary_identity_findings() -> list[str]:
    patterns = {
        "absolute macOS path": re.compile(rb"/Users/[^/\s]+/"),
        "non-anonymous PDF author metadata": re.compile(
            rb"\\Author\s*\((?!Anonymous(?:\s+Author(?:s)?|\s+Authors)?)",
            re.I,
        ),
        "email address": re.compile(
            rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I
        ),
    }
    if PUBLIC_MODE:
        # Public authorship is permitted. Credentials and local paths remain
        # prohibited, and the code/data boundary checks below stay in force.
        patterns.pop("non-anonymous PDF author metadata")
    findings: list[str] = []
    for path in tracked_files():
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
            continue
        content = path.read_bytes()
        for label, pattern in patterns.items():
            if pattern.search(content):
                findings.append(f"{label}: {path.relative_to(ROOT)}")
    return findings


def package_checks(aggregate_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = {
        "README.md",
        "LICENSE",
        "requirements.txt",
        "requirements-embedding.txt",
        "config/frozen_config.json",
        "config/identity_control_protocol.json",
        "data/README.md",
        "data/IDENTITY_MASKING_PROTOCOL.md",
        "data/DGS3MO_2023-10-01_2026-06-05.csv",
        "scripts/build_embeddings.py",
        "scripts/build_identity_controls.py",
        "scripts/build_identity_body_placebo.py",
        "scripts/build_pit_sector_labels.py",
        "scripts/build_targets.py",
        "scripts/build_manuscript_figures.py",
        "scripts/update_aggregate_manifest.py",
        "scripts/common.py",
        "scripts/compare_identity_controls.py",
        "scripts/compare_identity_control_ics.py",
        "scripts/identity_masking.py",
        "scripts/mask_identity_corpus.py",
        "scripts/portfolio.py",
        "scripts/recompute_published_results.py",
        "scripts/run_all.py",
        "scripts/run_identification_tests.py",
        "scripts/run_core.py",
        "scripts/callrank_features.py",
        "scripts/validate_canonical_ledger.py",
        "scripts/validate_release.py",
        "tests/test_public_code.py",
        "tests/test_identity_controls.py",
        "tests/test_frozen_identity_masker.py",
        "tests/test_section_parser.py",
    }
    if PUBLIC_MODE:
        required.update({"PUBLIC_RELEASE.json", "CODE_AND_DATA_NOTICE.md",
                         "requirements-validation.txt", "public_inputs/manifest.json",
                         "scripts/reproduce_public_inputs.py"})
    missing_code = [
        name for name in sorted(required) if not (ROOT / name).is_file()
    ]
    add(
        checks,
        "current source tree is complete",
        not missing_code,
        missing_code,
    )

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and is_public_path(path)
        and path.stat().st_size > MAX_FILE_BYTES
    ]
    add(checks, "public files are at most 8 MiB", not oversized, oversized)
    symlinks = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_symlink() and is_public_path(path)
    ]
    add(checks, "package contains no symbolic links", not symlinks, symlinks)
    empirical = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "data").rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".csv", ".parquet", ".pq", ".feather"}
        and path.name != "DGS3MO_2023-10-01_2026-06-05.csv"
    ]
    add(
        checks,
        "licensed input directory contains no private empirical rows",
        not empirical,
        empirical,
    )
    if PUBLIC_MODE:
        input_checks, approved_inputs = public_input_checks()
        checks.extend(input_checks)
        forbidden_binary_inputs = [str(path.relative_to(ROOT))
                                   for path in tracked_files()
                                   if path.suffix.lower() in {".parquet", ".pq", ".feather", ".npy", ".npz"}
                                   and path.resolve() not in approved_inputs]
        add(checks, "public tree contains no private empirical matrices", not forbidden_binary_inputs, forbidden_binary_inputs)
        metadata = public_release_metadata()
        add(checks, "named release metadata binds the public repository",
            metadata.get("release_mode") == "public_camera_ready"
            and metadata.get("repository_url") == "https://github.com/DoohwiCha/CallRank-ICAIF2026"
            and len(metadata.get("authors", [])) == 8
            and metadata.get("scientific_results_changed") is False)
    findings = text_findings()
    add(
        checks,
        "credential, path, and stale-version scan" if PUBLIC_MODE else "identity, credential, path, and stale-version scan",
        not findings,
        findings,
    )
    binary_findings = binary_identity_findings()
    add(
        checks,
        "binary metadata contains no email or local path" if PUBLIC_MODE else "binary metadata contains no identity, email, or local path",
        not binary_findings,
        binary_findings,
    )
    identities = git_identity_findings()
    add(
        checks,
        "reachable commits use declared public maintainer identity" if PUBLIC_MODE else "reachable commits use anonymous identity",
        not identities,
        identities,
    )
    history = git_history_shape()
    add(
        checks,
        "Git history has one public root on main" if PUBLIC_MODE else "Git history is one anonymous root on main",
        history["passed"],
        history["detail"],
    )

    config = load_config()
    add(
        checks,
        "artifact config uses the dated public release" if PUBLIC_MODE else "artifact config uses the dated review snapshot",
        str(config["version"]).startswith("camera-ready-fixed-head-") if PUBLIC_MODE else str(config["version"]).startswith("review-snapshot-"),
    )
    add(
        checks,
        "research boundary states the fixed-head temporal experiment",
        str(config["research_boundary"]).startswith(
            "Fixed-head temporal evaluation"
        ),
    )
    add(
        checks,
        "snapshot describes the reported fixed-head evidence",
        "reported fixed-head" in str(config["snapshot_status"])
        and "identity-controlled mechanism tests"
        in str(config["snapshot_status"]),
    )
    add(
        checks,
        "config makes no unsupported preselection claim",
        "portfolio rules remain fixed after the training boundary"
        not in str(config["research_boundary"])
        and "before firm-conditioned alternatives were evaluated"
        not in str(config["ranking_head"]["provenance"]),
    )
    add(checks, "raw text is excluded", config["raw_text_included"] is False)
    add(
        checks,
        "firm-level rows are excluded",
        config["firm_level_rows_included"] is False,
    )
    add(
        checks,
        "membership is point in time",
        config["universe"]["membership_is_point_in_time"] is True,
    )
    add(
        checks,
        "training outcomes end by the frozen boundary",
        config["ranking_head"]["latest_training_outcome"]
        == "2020-12-01",
    )
    add(
        checks,
        "ranking coefficients are not refit in evaluation",
        config["ranking_head"]["refit_after_training_boundary"] is False,
    )
    add(
        checks,
        "representations use separately fitted heads",
        config["ranking_head"]["shared_across_representations"] is False
        and str(config["ranking_head"]["fit_scope"]).startswith(
            "separate scaler"
        ),
    )
    add(
        checks,
        "signed forms have distinct construction roles without dominance claim",
        config["feature_definition"]["full_rank_signed_form"]
        == "raw signed residual"
        and config["feature_definition"]["bounded_allocation_signed_form"]
        == "L2-normalized signed residual"
        and config["feature_definition"]["empirical_dominance_claim"] is False,
    )
    add(
        checks,
        "firm history uses at least two prior observations",
        int(
            config["feature_definition"]["minimum_prior_firm_months"]
        )
        == 2,
    )
    add(
        checks,
        "temporal null uses 20,000 shifts",
        int(config["evaluation"]["circular_shift_randomizations"])
        == 20_000,
    )
    add(
        checks,
        "config records at most three equal-capital active sleeves",
        int(config["portfolio"]["maximum_concurrent_sleeves"]) == 3
        and str(config["portfolio"]["active_sleeve_weighting"])
        == "equal capital across overlapping sleeves",
    )
    identity_protocol_path = ROOT / "config" / "identity_control_protocol.json"
    identity_protocol = json.loads(identity_protocol_path.read_text(encoding="utf-8"))
    add(
        checks,
        "identity protocol transforms training and evaluation without returns",
        identity_protocol["head"]["train_and_evaluation_are_both_transformed"]
        is True
        and identity_protocol["same_identity_shell_body_swap"][
            "return_data_used_for_donor_mapping"
        ]
        is False
        and identity_protocol["public_release"]["aggregate_only"] is True,
    )
    identity_engine = ROOT / identity_protocol["full_pipeline_identity_mask"]["engine"]
    add(
        checks,
        "frozen identity masker source matches the declared engine hash",
        identity_engine.is_file()
        and sha256(identity_engine)
        == identity_protocol["full_pipeline_identity_mask"]["engine_sha256"],
        (
            sha256(identity_engine)
            if identity_engine.is_file()
            else "missing identity masker"
        ),
    )
    add(
        checks,
        "identity protocol fixes cap order, donor assignment, and timing seed",
        identity_protocol["text_construction"]["pre_mask_whitespace_normalization"]
        is False
        and identity_protocol["text_construction"][
            "identity_shell_composite_recapped"
        ]
        is False
        and identity_protocol["same_identity_shell_body_swap"][
            "assignment_prefix"
        ]
        == "callrank-identity-placebo-v1"
        and int(
            identity_protocol["same_identity_shell_body_swap"][
                "assignment_seed"
            ]
        )
        == 20260804
        and int(identity_protocol["evaluation"]["calendar_shift_random_seed"])
        == 20260804,
    )
    embedding_requirements = (ROOT / "requirements-embedding.txt").read_text(
        encoding="utf-8"
    )
    add(
        checks,
        "identity and embedding runtimes are exactly pinned",
        "sentence-transformers==5.6.0" in embedding_requirements
        and "torch==2.12.1" in embedding_requirements
        and "spacy==3.8.14" in embedding_requirements
        and "en_core_web_sm-3.8.0" in embedding_requirements,
    )
    identity_builder = (ROOT / "scripts" / "build_identity_controls.py").read_text(
        encoding="utf-8"
    )
    embedding_builder = (ROOT / "scripts" / "build_embeddings.py").read_text(
        encoding="utf-8"
    )
    core_runner = (ROOT / "scripts" / "run_core.py").read_text(encoding="utf-8")
    add(
        checks,
        "reported identity rebuild exposes frozen, no-recap, and fixed-seed paths",
        "--frozen-inputs" in identity_builder
        and "--preprocessed-identity-control" in embedding_builder
        and "--random-seed" in core_runner,
    )

    missing_aggregates = [
        name
        for name in sorted(AGGREGATE_REQUIRED)
        if not (aggregate_dir / name).is_file()
    ]
    add(
        checks,
        "aggregate result package is complete",
        not missing_aggregates,
        missing_aggregates,
    )
    manifest_path = aggregate_dir / "aggregate_manifest.json"
    add(checks, "aggregate manifest exists", manifest_path.is_file())
    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        bad_hashes = [
            name
            for name, metadata in manifest.get("files", {}).items()
            if not (aggregate_dir / name).is_file()
            or sha256(aggregate_dir / name) != metadata["sha256"]
        ]
        missing_manifest_entries = sorted(
            AGGREGATE_REQUIRED.difference(
                manifest.get("files", {}).keys()
            )
        )
        actual_manifest_entries = {
            str(path.relative_to(aggregate_dir))
            for path in aggregate_dir.rglob("*")
            if path.is_file()
            and path != manifest_path
            and not any(
                part.startswith(".")
                for part in path.relative_to(aggregate_dir).parts
            )
        }
        unexpected_manifest_entries = sorted(
            set(manifest.get("files", {})).difference(actual_manifest_entries)
        )
        unmanifested_files = sorted(
            actual_manifest_entries.difference(manifest.get("files", {}))
        )
        add(
            checks,
            "aggregate manifest hashes match",
            not bad_hashes
            and not missing_manifest_entries
            and not unexpected_manifest_entries
            and not unmanifested_files,
            {
                "bad_hashes": bad_hashes,
                "missing_entries": missing_manifest_entries,
                "unexpected_entries": unexpected_manifest_entries,
                "unmanifested_files": unmanifested_files,
            },
        )
    removed_aggregate_files = {
        "portfolio_breadth.csv",
        "portfolio_weighting.csv",
        "small_change_and_consensus_robustness.csv",
        "top_breadth_summary.csv",
        "training_period_portfolio_diagnostics.csv",
    }
    add(
        checks,
        "removed portfolio-width files are absent",
        not any((aggregate_dir / name).exists() for name in removed_aggregate_files),
        sorted(
            name
            for name in removed_aggregate_files
            if (aggregate_dir / name).exists()
        ),
    )
    if not missing_aggregates:
        portfolio_daily = pd.read_csv(
            aggregate_dir / "portfolio_daily_returns.csv"
        )
        top1_daily = portfolio_daily[
            portfolio_daily["strategy"].eq("top1")
        ]
        active = pd.to_numeric(
            top1_daily["active_sleeves"], errors="coerce"
        )
        maximum_active = int(active.max()) if active.notna().any() else -1
        add(
            checks,
            "released Top 1 path has at most three concurrent sleeves",
            not top1_daily.empty
            and active.notna().all()
            and active.between(0, 3).all()
            and maximum_active == 3,
            {
                "minimum_active_sleeves": (
                    int(active.min()) if active.notna().any() else None
                ),
                "maximum_active_sleeves": maximum_active,
            },
        )
        reproduction = verify_aggregate(aggregate_dir)
        add(
            checks,
            "published aggregate metrics recompute",
            reproduction["status"] == "PASS",
            {
                "maximum_numeric_error": reproduction[
                    "maximum_numeric_error"
                ],
                "checks": reproduction["checks"],
            },
        )
    if PUBLIC_MODE:
        valid_inputs = all(check["passed"] for check in input_checks)
        replay_detail: dict[str, Any] = {}
        replay_passed = False
        if valid_inputs:
            try:
                execution = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "scripts/reproduce_public_inputs.py")],
                    cwd=ROOT, capture_output=True, text=True, timeout=300,
                )
                replay = json.loads(execution.stdout)
                replay_passed = execution.returncode == 0 and replay.get("status") == "PASS"
                comparisons = replay.get("published_comparison", [])
                replay_detail = {
                    "refitted_models": len(replay.get("refitted_models", [])),
                    "training_rows": replay.get("training_rows"),
                    "evaluation_sector_months": replay.get("evaluation_sector_months"),
                    "numeric_comparisons": len(comparisons),
                    "maximum_numeric_error": max((row["absolute_error"] for row in comparisons), default=None),
                }
            except (subprocess.SubprocessError, ValueError, KeyError, OSError) as error:
                replay_detail = {"error_type": type(error).__name__}
        else:
            replay_detail = {"reason": "unverified replay inputs"}
        add(checks, "public sufficient-statistic heads and portfolios independently refit", replay_passed, replay_detail)
    return checks


def generated_checks(results_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required = {
        "representation_results.csv",
        "paired_ic_tests.csv",
        "representation_circular_shift_tests.csv",
        "fit_audit.csv",
        "sector_month_audit.csv",
        "temporal_circular_shift_null.csv",
        "run_summary.json",
        "row_level/resolution_predictions.parquet",
        "row_level/multiresolution_predictions.parquet",
        "row_level/unit_direction_daily_returns.parquet",
    }
    missing = [
        name for name in sorted(required) if not (results_dir / name).is_file()
    ]
    add(checks, "fresh run produced all core outputs", not missing, missing)
    if missing:
        return checks

    predictions = pd.read_parquet(
        results_dir / "row_level" / "multiresolution_predictions.parquet"
    )
    for column in ("month", "entry_date", "exit_date"):
        predictions[column] = pd.to_datetime(predictions[column])
    main_predictions = predictions[
        predictions["candidate"].eq(RAW_DISPLACEMENT)
    ]
    rank = rank_summary(main_predictions)
    reported = pd.read_csv(
        results_dir / "representation_results.csv"
    )
    main = reported[
        reported["representation"].eq(RAW_DISPLACEMENT)
        & reported["resolution"].eq("multiresolution")
        & reported["strategy"].eq("top1")
    ].iloc[0]
    add(
        checks,
        "fresh full-rank IC recomputes from predictions",
        abs(float(rank["rank_ic"]) - float(main["rank_ic"])) < 1e-10,
        abs(float(rank["rank_ic"]) - float(main["rank_ic"])),
    )

    daily = pd.read_parquet(
        results_dir / "row_level" / "unit_direction_daily_returns.parquet"
    )
    risk_free = pd.read_csv(
        ROOT / "data" / "DGS3MO_2023-10-01_2026-06-05.csv"
    )
    fields = (
        "annualized_return",
        "spy_return",
        "spy_alpha",
        "spy_alpha_p",
        "sharpe",
        "mdd",
    )
    errors: dict[str, float] = {}
    for strategy, frame in daily.groupby("strategy", sort=True):
        summary = strategy_summary(frame, risk_free=risk_free)
        row = reported[
            reported["representation"].eq(UNIT_DIRECTION)
            & reported["resolution"].eq("multiresolution")
            & reported["strategy"].eq(strategy)
        ].iloc[0]
        for field in fields:
            errors[f"{strategy}:{field}"] = abs(
                float(summary[field]) - float(row[field])
            )
    add(
        checks,
        "fresh economics recompute from daily returns",
        max(errors.values()) < 1e-10,
        max(errors.values()),
    )
    circular = pd.read_csv(
        results_dir / "representation_circular_shift_tests.csv"
    )
    null = pd.read_csv(
        results_dir / "temporal_circular_shift_null.csv"
    )
    main_circular = circular[
        circular["candidate"].eq(RAW_DISPLACEMENT)
    ].iloc[0]
    pvalue = float(
        (1 + null["rank_ic"].ge(main_circular["observed_rank_ic"]).sum())
        / (len(null) + 1)
    )
    add(
        checks,
        "fresh temporal p-value recomputes from null draws",
        len(null) == int(main_circular["randomizations"])
        and abs(pvalue - float(main_circular["upper_tail_p"])) < 1e-12,
        {
            "draws": len(null),
            "pvalue_error": abs(
                pvalue - float(main_circular["upper_tail_p"])
            ),
        },
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=ROOT / "aggregate_results",
    )
    args = parser.parse_args()
    checks = package_checks(args.aggregate_dir.resolve())
    scope = "public package"
    if args.results_dir:
        checks.extend(generated_checks(args.results_dir.resolve()))
        scope = "public package and fresh licensed-input run"
    failed = [check for check in checks if not check["passed"]]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        detail = check["detail"]
        suffix = (
            f" [{detail}]" if detail not in ("", None, [], {}) else ""
        )
        print(f"{status}: {check['check']}{suffix}")
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "scope": scope,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "checks": checks,
    }
    if args.results_dir:
        save_json(
            args.results_dir.resolve() / "generated_validation.json",
            summary,
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
