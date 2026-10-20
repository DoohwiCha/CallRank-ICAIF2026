# Code and data boundary

The MIT license applies to the source code in this repository. It grants no
rights to transcripts, historical membership, capitalization, firm embeddings,
or third-party market data. Raw calls, firm embeddings, issuer-level rows,
membership and capitalization rows, and original dollar-price quotes are
excluded. All portfolio paths are historical backtests. Upstream
reconstruction requires conforming licensed inputs under the relevant provider
agreements.

The released empirical materials are strategy-level aggregate tables and
paths, figures, public factor and risk-free series, and the explicitly scoped
inputs in `public_inputs/`. Those inputs comprise sector-level standardized
training Gram matrices, evaluation-to-training cross-Gram matrices, sector-month
keys and targets, and normalized sector-ETF/SPY total-return indices. They
permit independent fitting of the six raw-displacement and unit-direction
PCA/Ridge heads and recomputation of their rank ICs, Top 1, Bottom 1, and
spread allocations, portfolio statistics, and alpha, together with the SPY,
1/N, momentum, and low-volatility baselines.

The public replay begins at the sufficient-statistic boundary. Recomputing the
encoder features and firm-conditioned inputs requires the licensed transcripts.
The published feature statistics are sector-level inner products with
sector-month keys and targets.

Public factor observations in `aggregate_results/factor_daily_ff5_momentum.csv`
and `public_inputs/daily_factors.csv` come from the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
(accessed July 26, 2026). The underlying factor data retain the rights and
terms of Eugene F. Fama and Kenneth R. French; this repository does not license
them under MIT. The DGS3MO series in `data/` and `public_inputs/DGS3MO.csv` is
published by the Board of Governors of the Federal Reserve System and was
obtained from [FRED](https://fred.stlouisfed.org/series/DGS3MO) (accessed
July 25, 2026). Cite the original data sources when using these series.

`public_inputs/sector_total_return_indices.csv` holds the Yahoo Finance
dividend-adjusted closes of SPY and the eleven sector ETFs, each divided by its
first available value. The rebasing preserves every return and price ratio, so
the file carries the same information as the source series up to one scale
factor per ETF. The 42-trading-day targets in `public_inputs/` and
`diagnostics/` and the daily SPY and strategy returns in `aggregate_results/`
are derived from these series. These derived market data are not licensed under
MIT. They are included only so that readers can replicate the reported results,
and they remain subject to Yahoo's
[terms of service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
and those of its data providers; for any other use, obtain the series from the
source. Rights holders can request removal by opening an issue in this
repository.

The public checks recompute statistics from the released aggregate strategy
paths and randomization distributions, and check stored aggregate estimates
and hashes. `scripts/reproduce_public_inputs.py` additionally refits the heads
and recomputes ranking and allocation from public sufficient statistics.
Transcript-to-feature reconstruction requires the conforming licensed inputs
described in `data/README.md`.

The separate `diagnostics/unsigned_displacement_20261011/` addendum contains
sector-level model scores and targets, aggregate monthly ICs, paired statistics,
and calendar-shift distributions for a known-period post-review diagnostic.
These are derived research outputs: the sector scores are computed from the
licensed transcripts, and the targets from the market series above. They are
included only so that readers can replicate the reported diagnostic. Its replay
recalculates statistics from the released scores; it does not refit the
unsigned heads or reconstruct licensed firm-level features. The six-head
sufficient-statistic refit from `public_inputs/` is a separate workflow.
