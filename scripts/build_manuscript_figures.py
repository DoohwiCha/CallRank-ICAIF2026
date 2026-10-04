#!/usr/bin/env python3
"""Rebuild the two manuscript diagnostic figures from released aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "primary": "#2C6E63",
    "secondary": "#98A2A8",
    "benchmark": "#6B737B",
    "grid": "#DDE2E5",
    "axis": "#7D878E",
    "text": "#3E4A52",
}
PDF_METADATA = {
    "Creator": "CallRank manuscript figure builder",
    "CreationDate": None,
    "ModDate": None,
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        metadata=PDF_METADATA,
    )
    fig.savefig(
        stem.with_suffix(".png"),
        bbox_inches="tight",
        pad_inches=0.03,
        dpi=180,
    )


def robustness_figure(aggregate: Path, figures: Path) -> None:
    plot = pd.read_csv(aggregate / "raw_robustness.csv").iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.55,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.48, 2.76))
        ax.axvline(0.0, color=COLORS["axis"], linewidth=0.8, linestyle=(0, (3, 2)))
        for index, item in plot.iterrows():
            primary = item["key"] == "main"
            ax.hlines(
                index,
                item["low"],
                item["high"],
                color=COLORS["primary"] if primary else COLORS["axis"],
                linewidth=1.5 if primary else 1.05,
            )
            ax.scatter(
                item["ic"],
                index,
                s=26 if primary else 20,
                marker="D" if primary else "o",
                facecolor=COLORS["primary"],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.text(
                item["high"] + 0.006,
                index,
                f"{item['ic']:.3f}".lstrip("0"),
                ha="left",
                va="center",
                fontsize=6.2,
            )
        ax.set_yticks(y, plot["label"])
        ax.set_xlim(-0.045, 0.31)
        ax.set_xticks((0.0, 0.1, 0.2, 0.3))
        ax.set_xlabel("Mean monthly raw-displacement rank IC")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=2.2, width=0.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_linewidth(0.55)
        fig.subplots_adjust(left=0.42, right=0.985, top=0.98, bottom=0.20)
        save(fig, figures / "callrank_robustness")
        plt.close(fig)


def history_figure(aggregate: Path, figures: Path) -> None:
    history = pd.read_csv(aggregate / "history_reference_tests.csv").set_index("test")
    identity = pd.read_csv(aggregate / "issuer_identity_probe.csv").set_index("representation")
    draws = pd.read_csv(aggregate / "wrong_firm_placebo_draws.csv")
    own = float(history.loc["own_firm_history", "mean_ic"])
    own_p = float(history.loc["own_firm_history", "newey_west_p"])
    wrong = float(history.loc["same_sector_wrong_firm_placebo_mean", "mean_ic"])
    placebo_p = float((1 + draws["mean_rank_ic"].ge(own).sum()) / (len(draws) + 1))
    current = 100.0 * float(identity.loc["current_qa", "top1_accuracy"])
    direction = 100.0 * float(identity.loc["own_history_direction", "top1_accuracy"])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.55,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(3.48, 2.55),
            gridspec_kw={"height_ratios": (1.0, 1.0), "hspace": 0.74},
        )
        history_values = np.asarray([own, wrong])
        axes[0].barh(
            np.arange(2),
            history_values,
            height=0.50,
            color=(COLORS["primary"], COLORS["secondary"]),
            edgecolor="none",
            zorder=3,
        )
        axes[0].set_yticks(np.arange(2), ("Own-firm history", "Wrong firm, same sector"))
        axes[0].invert_yaxis()
        axes[0].set_xlim(0.0, 0.21)
        axes[0].set_xticks((0.0, 0.05, 0.10, 0.15, 0.20))
        axes[0].set_xlabel("Mean monthly rank IC", labelpad=2)
        axes[0].set_title(f"(a) Matched history placebo ($p$={placebo_p:.3f})", loc="left", pad=3)
        annotations = (f"{own:.3f} [NW $p$={own_p:.3f}]", f"{wrong:.3f} [mean]")
        for position, (value, annotation) in enumerate(zip(history_values, annotations, strict=True)):
            axes[0].text(value + 0.004, position, annotation.lstrip("0"), ha="left", va="center", fontsize=6.2)

        identity_values = np.asarray([current, direction])
        axes[1].barh(
            np.arange(2),
            identity_values,
            height=0.50,
            color=(COLORS["benchmark"], COLORS["primary"]),
            edgecolor="none",
            zorder=3,
        )
        axes[1].set_yticks(np.arange(2), ("Current Q&A", "Own-history direction"))
        axes[1].invert_yaxis()
        axes[1].set_xlim(0.0, 46.0)
        axes[1].set_xticks((0.0, 10.0, 20.0, 30.0, 40.0))
        axes[1].set_xlabel("Issuer top-1 accuracy (%)", labelpad=2)
        axes[1].set_title("(b) Issuer-identity retrieval", loc="left", pad=3)
        for position, value in enumerate(identity_values):
            axes[1].text(value + 1.0, position, f"{value:.1f}%", ha="left", va="center", fontsize=6.2)
        for ax in axes:
            ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, zorder=0)
            ax.tick_params(axis="y", length=0, pad=3)
            ax.tick_params(axis="x", length=2.2, width=0.5)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(COLORS["axis"])
            ax.spines["bottom"].set_linewidth(0.55)
        fig.subplots_adjust(left=0.40, right=0.985, top=0.98, bottom=0.12)
        save(fig, figures / "callrank_history_reference")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=ROOT / "aggregate_results")
    args = parser.parse_args()
    aggregate = args.aggregate_dir.resolve()
    figures = aggregate / "figures"
    robustness_figure(aggregate, figures)
    history_figure(aggregate, figures)
    print(f"Rebuilt manuscript diagnostic figures in {figures}")


if __name__ == "__main__":
    main()
