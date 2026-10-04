# Aggregate Results

This directory contains manuscript-level results from the clean-room call
ledger. It excludes transcripts, call-level text, embeddings, predictions,
holdings, membership rows, capitalization rows, and provider price histories.

`canonical_data_summary.json` and `cleanroom_integrity_summary.json` record the
final population, return-blind corrections, and immutable ledger hashes.
`representation_results.csv`, `paired_ic_tests.csv`,
`circular_shift_tests.csv`, and `temporal_circular_shift_null.csv` support the
main signed-residual-versus-distance result. The remaining tables support
financial controls, data sensitivities, fixed-head diagnostics, encoder
comparisons, matched economic baselines, tie handling, long-lag alpha
inference, concentration diagnostics, and cost-adjusted allocation.
`cost_sensitivity.csv` reapplies one-way costs from 5 to 20 basis points to the
released gross paths and realized turnover.
`raw_portfolio_summary.csv`, `raw_portfolio_daily_returns.csv`, and
`raw_cost_sensitivity.csv` reproduce the economic columns for the raw signed
residual in the manuscript's signed-form comparison.
`history_reference_tests.csv`,
`wrong_firm_placebo_draws.csv`, and `issuer_identity_probe.csv` support the
unit-direction firm-matching and issuer-persistence probe. The history table
contains the own-firm result, same-sector alternatives, and wrong-firm placebo
mean. `raw_robustness.csv` is the source for every row in the manuscript's
raw-displacement robustness figure. `structured_news_tests.csv` retains the
separate unit-direction reported-earnings diagnostic and is not the source of
that figure.

`identity_control_tests.csv` contains the eight aggregate estimates in the
manuscript's identity-control table: two signed forms for the full-pipeline
identity mask, correct masked body, wrong-body donor, and paired
correct-minus-donor gap. `identity_control_monthly_ic.csv` releases the 24
monthly aggregate ICs for each arm and signed form, while
`identity_control_shift_null.csv` releases the 20,000 mask-pipeline timing
draws for each signed form. These files reproduce the reported means,
Newey--West tests, paired gaps, and mask-pipeline shift p-values without
revealing a firm or sector row. The licensed-text transformation is specified
in `../config/identity_control_protocol.json`; its frozen masking engine and
licensed-input boundary are documented in `../scripts/identity_masking.py` and
`../data/IDENTITY_MASKING_PROTOCOL.md`. The public monthly and null files are
the exact aggregate replay. No masked text, donor mapping, firm-level
embedding, prediction, or holding row is released.

`leave_one_sector_out.csv` and `grouped_signal_month_exclusions.csv` report
aggregate concentration diagnostics. `training_correction_ablation.csv`
records the effect of restoring the omitted issuer and removing the two
wrong-issuer training calls. `static_sector_benchmark.csv` reports matched-date
SPY and XLU buy-and-hold return and drawdown without redistributing price
histories.

`portfolio_daily_returns.csv` contains aggregate Top 1, Bottom 1, spread,
tie-sensitivity, and adverse-check paths for unit direction.
`raw_portfolio_daily_returns.csv` contains the corresponding raw-displacement
paths, and `baseline_portfolio_daily_returns.csv` contains the four matched
baseline paths. The `active_sleeves` column records that zero to three monthly
sleeves can be active on a date; aggregate capital is divided equally across
the active sleeves. `random_sector_null.csv` contains 2,000 path-level summary
metrics, not sector selections. These files contain no sector holding,
prediction, company, transcript, membership, capitalization, or provider-price
row.
