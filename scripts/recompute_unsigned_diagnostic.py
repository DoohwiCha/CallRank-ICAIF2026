#!/usr/bin/env python3
"""Replay the post-review unsigned diagnostic from public sector scores.

This recalculates score ensembles, monthly ICs, paired inference and calendar
shifts. It does not refit unsigned models or reconstruct licensed firm features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "diagnostics" / "unsigned_displacement_20261011"
KEYS = ["candidate", "sector", "month"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    # Exact score ties affect Spearman ranks and rotations; round-trip parsing
    # preserves the float64 values emitted by the original diagnostic.
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def monthly_ic(frame: pd.DataFrame, minimum: int) -> pd.DataFrame:
    rows = []
    for month, group in frame.groupby("month", sort=True):
        if len(group) >= minimum:
            rows.append({"month": month, "sectors": len(group),
                         "ic": float(stats.spearmanr(group.pred, group.excess_return).statistic)})
    return pd.DataFrame(rows)


def hac(values, lag: int) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    fit = sm.OLS(clean, np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag})
    return {f"nw{lag}_se": float(fit.bse[0]), f"nw{lag}_p": float(fit.pvalues[0]),
            f"nw{lag}_ci_low": float(fit.conf_int()[0, 0]),
            f"nw{lag}_ci_high": float(fit.conf_int()[0, 1])}


def rotate_scores(frame, draws, seed, minimum):
    """Original nonzero sector shifts, batched without changing RNG order."""
    source = frame.sort_values(["sector", "month"]).reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in source.groupby("sector", sort=True) if len(g) > 1]
    months = [g.index.to_numpy() for _, g in source.groupby("month", sort=True) if len(g) >= minimum]
    targets = [stats.rankdata(source.iloc[pos].excess_return.to_numpy()) for pos in months]
    targets = [rank - rank.mean() for rank in targets]
    denominators = [np.linalg.norm(rank) for rank in targets]
    values = source.pred.to_numpy()
    rng = np.random.default_rng(seed)
    null = []
    for begin in range(0, draws, 250):
        size = min(250, draws - begin)
        offsets = np.array([[rng.integers(1, len(pos)) for pos in groups] for _ in range(size)])
        shifted = np.tile(values, (size, 1))
        for column, pos in enumerate(groups):
            rolled = (np.arange(len(pos))[None, :] - offsets[:, column, None]) % len(pos)
            shifted[:, pos] = values[pos[rolled]]
        means = np.zeros(size)
        for pos, target, denominator in zip(months, targets, denominators):
            ranks = stats.rankdata(shifted[:, pos], axis=1)
            ranks -= ranks.mean(axis=1, keepdims=True)
            means += (ranks @ target) / (np.linalg.norm(ranks, axis=1) * denominator)
        null.extend(means / len(months))
    return np.asarray(null)


def compare_frames(actual, expected, keys, label, errors):
    merged = actual.merge(expected, on=keys, how="outer", suffixes=("_new", "_reported"), indicator=True,
                          validate="one_to_one")
    if not merged._merge.eq("both").all():
        raise ValueError(f"{label}: key sets differ")
    for column in expected:
        if column in keys or not pd.api.types.is_numeric_dtype(expected[column]):
            continue
        if column not in actual:
            raise ValueError(f"{label}: missing recomputed {column}")
        a = merged[column + "_new"].to_numpy(dtype=float)
        b = merged[column + "_reported"].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(f"{label}:{column}: missingness differs")
        error = float(np.max(np.abs(a[np.isfinite(b)] - b[np.isfinite(b)]), initial=0.0))
        errors[f"{label}:{column}"] = error


def verify(directory: Path, tolerance: float) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    checksum_checks = {}
    for relative, item in manifest["files"].items():
        path = ROOT / relative
        checksum_checks[relative] = path.is_file() and digest(path) == item["sha256"] and path.stat().st_size == item["bytes"]
    if not all(checksum_checks.values()):
        raise ValueError("Diagnostic file checksum mismatch")
    protocol = json.loads((directory / "protocol.json").read_text())
    minimum = protocol["evaluation"]["min_sectors"]
    shift = protocol["inference"]["shift"]
    by_chunk = read_csv(directory / "predictions_by_chunk.csv")
    expected_arms = set(protocol["arms"])
    if set(by_chunk.candidate) != expected_arms:
        raise ValueError("Missing or unexpected diagnostic arm")
    if set(by_chunk.chunk_words) != set(protocol["features"]["words"]):
        raise ValueError("Unexpected passage length")
    if by_chunk.duplicated(["chunk_words", *KEYS]).any():
        raise ValueError("Duplicate prediction key")
    allowed_sectors = set(read_csv(ROOT / "public_inputs" / "eval_keys.csv").sector)
    if not set(by_chunk.sector).issubset(allowed_sectors):
        raise ValueError("Unexpected sector label")
    by_chunk["pred_rank"] = by_chunk.groupby(["candidate", "chunk_words", "month"]).pred.rank(method="average", pct=True)
    ensemble = by_chunk.groupby(KEYS, sort=True).agg(
        entry_date=("entry_date", "first"), exit_date=("exit_date", "first"),
        excess_return=("excess_return", "first"), pred=("pred_rank", "mean"),
        chunk_models=("chunk_words", "nunique")).reset_index()
    if not ensemble.chunk_models.eq(3).all():
        raise ValueError("An ensemble lacks a passage model")
    errors = {}
    compare_frames(ensemble, read_csv(directory / "ensemble_predictions.csv"), KEYS, "ensemble", errors)
    summaries, monthly_rows = [], []
    for (candidate, word), frame in by_chunk.groupby(["candidate", "chunk_words"]):
        ic = monthly_ic(frame, minimum).ic
        summaries.append({"candidate": candidate, "resolution": str(word), "months": len(ic),
                          "ic": float(ic.mean()), **hac(ic, 1), **hac(ic, 2)})
    for candidate, frame in ensemble.groupby("candidate"):
        monthly = monthly_ic(frame, minimum)
        monthly.insert(0, "candidate", candidate)
        monthly_rows.append(monthly)
        null = rotate_scores(frame, shift["draws"], shift["seed"], minimum)
        expected_null = read_csv(directory / f"shift_null_{candidate}.csv").ic.to_numpy()
        if len(null) != len(expected_null):
            raise ValueError("Shift path count differs")
        errors[f"shift_paths:{candidate}"] = float(np.max(np.abs(null - expected_null)))
        observed = float(monthly.ic.mean())
        summaries.append({"candidate": candidate, "resolution": "ensemble", "months": len(monthly),
                          "ic": observed, **hac(monthly.ic, 1), **hac(monthly.ic, 2),
                          "shift_draws": shift["draws"], "shift_seed": shift["seed"],
                          "shift_null_mean": float(null.mean()), "shift_null_sd": float(null.std(ddof=1)),
                          "shift_p": float((1 + np.sum(null >= observed)) / (shift["draws"] + 1))})
    monthly_all = pd.concat(monthly_rows, ignore_index=True)
    compare_frames(monthly_all, read_csv(directory / "monthly_ics.csv"), ["candidate", "month"], "monthly_ic", errors)
    compare_frames(pd.DataFrame(summaries), read_csv(directory / "rank_summary.csv", dtype={"resolution": str}),
                   ["candidate", "resolution"], "rank_summary", errors)
    angular = read_csv(directory / "angular_monthly_ic.csv")
    wide = pd.concat([monthly_all, angular], ignore_index=True).pivot(index="month", columns="candidate", values="ic")
    pairs = []
    for left, right in protocol["inference"]["paired_tests"]:
        difference = wide[left] - wide[right]
        pairs.append({"left": left, "right": right, "months": len(difference),
                      "mean_difference": float(difference.mean()), **hac(difference, 1), **hac(difference, 2),
                      "paired_t_p": float(stats.ttest_1samp(difference, 0).pvalue)})
    compare_frames(pd.DataFrame(pairs), read_csv(directory / "paired_ic_tests.csv"),
                   ["left", "right"], "paired_ic", errors)
    maximum = max(errors.values(), default=0.0)
    return {"status": "PASS" if maximum <= tolerance else "FAIL",
            "boundary": "Known-period post-review score replay; no unsigned-head refit or licensed-feature reconstruction.",
            "p_values": "unadjusted; NW1 primary convention, NW2 and conventional paired t sensitivities retained",
            "checksum_files": len(checksum_checks), "checksums_pass": all(checksum_checks.values()),
            "prediction_rows": len(by_chunk), "ensemble_rows": len(ensemble),
            "arms": len(expected_arms), "passage_lengths": 3, "shift_paths_recomputed": shift["draws"] * len(expected_arms),
            "numeric_comparisons": len(errors), "maximum_numeric_error": maximum,
            "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, help="Optional JSON validation receipt")
    args = parser.parse_args()
    result = verify(args.diagnostic_dir, args.tolerance)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
