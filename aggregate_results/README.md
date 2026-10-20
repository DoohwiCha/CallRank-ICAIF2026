# Aggregate results

This directory holds the paper-level results built from the clean-room call
ledger: aggregate tables, daily strategy paths, randomization distributions,
and the paper figures. It contains no transcript, call-level text, embedding,
firm-level prediction, holding, membership, or capitalization row, and no
provider price level. The daily paths include SPY returns and strategy returns
computed from the Yahoo Finance dividend-adjusted series described in
[`CODE_AND_DATA_NOTICE.md`](../CODE_AND_DATA_NOTICE.md); these derived market
data are not licensed under MIT. All portfolio results are historical
backtests.

## Files by paper table and figure

| Paper | Files |
| --- | --- |
| Table 1 and §5.1 (data) | `canonical_data_summary.json`, `cleanroom_integrity_summary.json` (key map below) |
| Table 2, Panel A (representations) | `representation_results.csv`, `circular_shift_tests.csv`, `temporal_circular_shift_null.csv`, `paired_ic_tests.csv`; the unsigned rows are in [`diagnostics/unsigned_displacement_20261011/`](../diagnostics/unsigned_displacement_20261011/) |
| Table 2, Panel B (encoders; unit direction, 160-word passages) | `encoder_summary.csv` |
| Table 3 (identity controls) | `identity_control_tests.csv`, `identity_control_monthly_ic.csv`, `identity_control_shift_null.csv` |
| Table 4, Panels A–B, and Figure 3 (unit-direction allocation) | `portfolio_summary.csv`, `baseline_portfolio_summary.csv`, `factor_regressions.csv`, `alpha_lag_sensitivity.csv`, `portfolio_daily_returns.csv`, `baseline_portfolio_daily_returns.csv`, `factor_daily_ff5_momentum.csv` |
| Table 4, Panel C (raw displacement) | `raw_portfolio_summary.csv`, `raw_portfolio_daily_returns.csv`, `raw_cost_sensitivity.csv` |
| Figure 2 (firm matching) | `history_reference_tests.csv`, `wrong_firm_placebo_draws.csv`, `issuer_identity_probe.csv`, `identification_expansion_summary.json` |
| Figure 4 (raw-displacement robustness) | `raw_robustness.csv` |
| §3.5 (head grid, unit direction) | `pca_ridge_sensitivity.csv` |
| §5.1 (ledger corrections) | `training_correction_ablation.csv` |
| §5.4 (costs) | `cost_sensitivity.csv` |
| §5.4 (random-sector null) | `random_sector_null.csv`, `random_sector_null_summary.json` |
| §5.4 (concentration and stress tests) | `top1_selection_frequency.csv`, `aggregate_summary.json`, `static_sector_benchmark.csv`, `leave_one_sector_out.csv`, `grouped_signal_month_exclusions.csv`, `portfolio_yearly.csv` |
| §5.5 (unit-direction firm-count partial IC, .073) | `financial_controls.csv` |
| Figures 1–4 | `figures/` (see below) |

## Labels used in the files

- **Representations.** `firm_conditioned_qa_raw` is raw displacement,
  `firm_conditioned_qa_direction` is unit direction, and
  `firm_conditioned_qa_distance` is angular distance. The other Table 2 rows
  are `qa_level` (current Q&A level), `prepared_level` (prepared-remarks
  level), `qa_and_history_concat` (current/history concatenation), and
  `same_call_section_direction` (same-call prepared/Q&A direction: the
  normalized difference between a sector-month's normalized mean current-Q&A
  vector and its normalized mean prepared-remarks vector).
  `same_call_firm_direction` averages each firm's normalized Q&A-minus-prepared
  vector instead and is not a Table 2 row.
- **Ledger corrections.** In `training_correction_ablation.csv`, `old` is the
  original ledger (.142 unit, .164 raw), `omitted_issuer_restored` adds the 34
  verified calls of the initially omitted issuer (.097, .159), and
  `full_correction` is the final ledger (.100, .158); `wrong_issuer_removed`
  removes only the two wrong-issuer training calls.
- **Placebo share.** In `history_reference_tests.csv`, the placebo row's
  `positive_month_share` (0.999) is the share of the 1,000 placebo draws with a
  positive mean IC, not a share of months.
- **Capitalization weights.** The prior-close capitalization weighting in
  `config/frozen_config.json` (`portfolio.weighting`,
  `portfolio.tie_sensitivity`) matters only when a sleeve holds several
  sectors, as in the Bottom 1 and spread tie rows (see
  [Exact ties](#exact-ties)) and the `scored_sector_cap_base` reference path.
  Each reported Top 1 sleeve holds its single selected sector ETF.
- **Figure 2 files.** In `history_reference_tests.csv`, `own_firm_history` is
  the own-firm reference and `same_sector_wrong_firm_placebo_mean` is the mean
  of the 1,000 wrong-firm placebo draws; the two
  `sector_history_excluding_firm*` rows replace the own-firm reference with the
  same sector's other firms and are not reported in the paper. In
  `issuer_identity_probe.csv`, `current_qa` and `own_history_direction` are
  Figure 2b's current Q&A and unit direction.
- **Table 3 files.** In the `identity_control_*` files, `raw_residual` is raw
  displacement and `unit_direction` is unit direction;
  `full_pipeline_identity_mask`, `correct_masked_body`, `wrong_body_donor`, and
  `paired_correct_minus_donor` are the full-pipeline identity mask, correct
  masked body, body-swap placebo, and paired correct–placebo gap rows.
- **Other labels.** In `encoder_summary.csv`, `sec_bert`, `minilm`, and `mpnet`
  are Industry-BERT-SEC, MiniLM, and MPNet (Table 2, Panel B).
  `lm_tone_residual_top1` in the baseline files is Table 4's LM Q&A tone
  residual. `top1_ex_utilities` in `portfolio_summary.csv` is the §5.4
  Utilities-replacement stress test (the same path as the Utilities row of
  `leave_one_sector_out.csv`).

## Table 1 sources

| Table 1 value | Source key |
| --- | --- |
| 22,897 parsed transcript records | `canonical_data_summary.json`: `source_rows` |
| 21,799 strict-date usable | `canonical_data_summary.json`: `strict_date_usable` |
| 21,595 identity/point-in-time confirmed before deduplication | `canonical_data_summary.json`: `canonical_before_dedup` |
| 21,460 canonical calls after duplicate removal | `canonical_data_summary.json`: `canonical_usable` (= `cleanroom_integrity_summary.json`: `population.canonical_preclean_usable`) |
| 21,421 final locked calls | `cleanroom_integrity_summary.json`: `population.cleanroom_usable` (= `config/frozen_config.json`: `data_integrity.canonical_usable_calls`) |
| 4,604 post-encoder-release retained calls | `cleanroom_integrity_summary.json`: `population.evaluation_cleanroom_usable` |
| 2,848 day-level externally verified / 1,756 month-only, filing-consistent | `cleanroom_integrity_summary.json`: `population.evaluation_call_date_exact` / `population.evaluation_call_date_month_confirmed` (= `config/frozen_config.json`: `data_integrity.evaluation_exact_official_date` / `data_integrity.evaluation_official_month`; the frozen key names predate the paper's labels) |
| 1,385 full-panel sector-month cells (1,358 complete 42-trading-day targets / 27 terminal incomplete-target cells) | `config/frozen_config.json`: `data_integrity.sector_month_rows` / `data_integrity.complete_targets` (27 = 1,385 − 1,358; not the 27 unresolved calls in `data_integrity.evaluation_unresolved_excluded`) |
| 292 census / 265 target-complete / 240 reported eligible sector-month cells | `cleanroom_integrity_summary.json`: `coverage_and_sections.evaluation_sector_months` / rows of `public_inputs/eval_keys.csv` / `sector_month_cells` in `sector_concentration_summary.csv` |
| 32 census call months / 29 target-complete call months / 24 reported eligible decision months | not stored (the census spans November 2023–June 2026; Table 1 notes) / distinct `month` values in `public_inputs/eval_keys.csv` / `aggregate_summary.json`: `evaluation_months` |

`canonical_release_calls` (4,628) in `canonical_data_summary.json` is a count
before the clean-room gate. `call_date_audit_summary.json` records the earlier
external call-date audit of the canonical ledger (4,823 audited calls: 2,946
day-level passes, 1,837 month-level passes, 37 review and 3 fail outcomes;
4,635 evaluation-period calls after the audit). Table 1's day-level and
month-only counts are the clean-room values above.

The participation statistics in `cleanroom_integrity_summary.json`
(`coverage_and_sections`: minimum 1, median 13, and 18 single-firm cells) count
every firm with a clean-room call in the 292 post-encoder-release census
sector-months. The paper's §5.1 coverage statement counts eligible contributing
firms in the 240 scored sector-months (median 14; 13 single-contributor cells);
those two values are computed from licensed firm-level rows and are not stored
here.

## Diagnostics that share names with paper tests

- `leave_one_sector_out.csv` removes one sector at a time from the fixed
  unit-direction ranking and Top 1 selection universe (heads are not refit).
  It is not the paper's held-out-sector model, whose raw-displacement estimate
  (.158) is the `held_out` row of `raw_robustness.csv`; its Utilities row is
  the Utilities-replacement stress test in §5.4 (29.3% annualized net return,
  11.2% SPY-adjusted alpha).
- `held_out_sector_tests.csv` and `financial_controls.csv` are unit-direction
  diagnostics (the `cleanroom_mapping` row of `financial_controls.csv` is the
  unit IC, .100). The paper's raw-displacement values in Figure 4 are all in
  `raw_robustness.csv`.
- `structured_news_tests.csv` is a separate unit-direction reported-earnings
  diagnostic and is not the source of Figure 4.
- `pca_ridge_sensitivity.csv` is the released unit-direction PCA/Ridge grid
  cited in §3.5 (8, 16, 32, and 64 components; penalties 10² to 10⁵).

## Other files

- `aggregate_summary.json` records run settings, input hashes, and the Top 1
  selection counts behind the ten-sector and 16-of-23-transition statements.
- `identification_expansion_summary.json` records sample sizes and settings for
  the Figure 2 tests.
- Repository diagnostics that the paper does not report:
  - `direction_concentration_summary.json` and `sector_concentration_summary.csv`
    (directional agreement of unit-direction firm vectors)
  - `factor_concentration_summary.json` (sector- and month-exclusion alpha
    summaries; its `leave_one_month.three_largest_alpha_contributor_months`
    lists the three signal months removed in the §5.4 stress test)
  - `portfolio_comparison_tests.csv` (HAC test of the mean daily Top 1 minus
    Bottom 1 net return)
  - the `ff5_mom_macro_alpha` columns of `grouped_signal_month_exclusions.csv`
    and `leave_one_sector_out.csv`, which add two FRED macro series (recorded by
    hash only) to the FF5+Momentum model
  - the `scored_sector_cap_base` reference path
- `identity_control_monthly_ic.csv` holds the 24 monthly ICs for each arm and
  signed form, and `identity_control_shift_null.csv` holds the 20,000
  calendar-shift draws of the full-pipeline identity mask for each signed
  form. They reproduce the reported means, Newey–West tests,
  paired gaps, and shift p-values without revealing a firm or sector row. The
  licensed-text transformation is specified in
  [`config/identity_control_protocol.json`](../config/identity_control_protocol.json),
  [`scripts/identity_masking.py`](../scripts/identity_masking.py), and
  [`data/IDENTITY_MASKING_PROTOCOL.md`](../data/IDENTITY_MASKING_PROTOCOL.md).
- `portfolio_daily_returns.csv` contains the unit-direction Top 1, Bottom 1,
  Top 1–Bottom 1 spread, all-cutoff tie (`*_all_cutoff_ties`; see
  [Exact ties](#exact-ties)), and adverse-check paths;
  `raw_portfolio_daily_returns.csv` and `baseline_portfolio_daily_returns.csv`
  contain the raw-displacement and matched-baseline paths. The `active_sleeves`
  column shows that zero to three monthly sleeves are active on a date, and
  capital is divided equally across active sleeves.
- `static_sector_benchmark.csv` reports matched-date SPY and XLU buy-and-hold
  returns and drawdowns.
- `random_sector_null.csv` holds 2,000 path-level summary metrics, not sector
  selections.

## Figures

`figures/` holds the four paper figures as PDF/PNG pairs:

| File | Figure |
| --- | --- |
| `callrank_method` | Figure 1 |
| `callrank_history_reference` | Figure 2 |
| `callrank_sector_allocation` | Figure 3 |
| `callrank_robustness` | Figure 4 |

The PDFs are byte-identical to the camera-ready figure files.
`scripts/build_manuscript_figures.py` rebuilds Figures 2 and 4 from the tables
above; Figures 1 and 3 are stored as released.

`callrank_time_alignment` is a supplementary figure, not a paper figure. It
shows the 20,000-path calendar-shift null for the raw-displacement mean monthly
sector-rank IC (observed .158; calendar-shift p = .044; Table 2), drawn from
`temporal_circular_shift_null.csv`.

## Exact ties

This section explains the paper's exact-tie sensitivity (§4.2) and the
released tie rows.

- **The tie.** In the April 2025 unit-direction Top 1 decision, the two highest
  ensemble scores are equal in exact arithmetic: each averages three
  within-month percentile ranks with the same sum. The fixed canonical
  sector-key order resolves this tie without using returns.
- **The paper's values.** Holding both tied sectors, weighted by prior-close
  capitalization, yields a 41.9% annualized net return and 26.9% SPY-adjusted
  alpha. Capitalization rows are licensed, so these two values are cited from
  the paper and are not stored here.
- **The released rows.** The `*_all_cutoff_ties` rows apply the same rule to
  the stored float64 scores. In the April 2025 decision, the two tied sectors'
  stored scores differ by one unit in the last place, and the higher one
  belongs to the sector that the canonical order selects, so
  `top1_all_cutoff_ties` equals the canonical Top 1 path (42.1%; 27.0%). No
  raw-displacement Top 1 decision is tied, so the raw `top1_all_cutoff_ties`
  row also equals its canonical row.
- **Bottom 1 ties.** In one decision for each signed form (unit direction, May
  2025; raw displacement, October 2025), the two lowest scores are equal in
  exact arithmetic and also in float64. The `bottom1_all_cutoff_ties` and
  `top1_minus_bottom1_all_cutoff_ties` rows hold both sectors in that sleeve,
  weighted by prior-close capitalization, so they differ from the canonical
  Bottom 1 and spread rows. The paper's tie sensitivity covers the Top 1
  decisions only.
- **Labels.** "Exact cutoff tie" in the rule labels, in
  `aggregate_summary.json`, and in `config/frozen_config.json` means equal
  float64 values.
- **Reading the scores.** The sector-level scores are released in
  [`diagnostics/unsigned_displacement_20261011/ensemble_predictions.csv`](../diagnostics/unsigned_displacement_20261011/ensemble_predictions.csv),
  and `scripts/reproduce_public_inputs.py --output-dir` recomputes them as
  `predictions.csv`. Read them with `float_precision="round_trip"` in pandas;
  default parsing can blur the one-unit difference.
