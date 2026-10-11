# Code and data boundary

The MIT license applies to the source code in this repository. It grants no
rights to transcripts, historical membership, capitalization, firm embeddings,
or third-party market data. Raw calls, firm embeddings, issuer identifiers,
membership and capitalization rows, original dollar-price quotes, and actual
account holdings are excluded. Upstream reconstruction requires conforming
licensed inputs under the relevant provider agreements.

The released empirical materials are strategy-level aggregate tables and
paths, figures, public factor and risk-free series, and the explicitly scoped
inputs in `public_inputs/`. Those inputs comprise sector-level standardized
training Gram matrices, evaluation-to-training cross-Gram matrices, sector-month
keys and targets, and normalized sector-ETF/SPY total-return indices. They
permit independent fitting of the PCA/Ridge head and recomputation of the
reported rank ICs, model allocations, portfolio statistics, and alpha.

The public replay begins at the sufficient-statistic boundary. Encoder and
firm-conditioned feature reconstruction require licensed upstream inputs.
The published feature statistics are sector-level inner products with
sector-month keys and targets.

Public factor observations in `aggregate_results/factor_daily_ff5_momentum.csv`
come from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The underlying factor data retain the rights and terms of Eugene F. Fama and
Kenneth R. French; this repository does not license them under MIT. The DGS3MO
series in `data/` comes from the Federal Reserve Board via
[FRED](https://fred.stlouisfed.org/series/DGS3MO). Cite the original data sources
when using either series.

The normalized total-return indices and sector targets derive from the study's
dividend-adjusted sector-ETF/SPY market series, sourced from Yahoo Finance.
They are derived research inputs and retain the source-specific rights and
terms of the underlying market observations. They are not licensed under MIT.

The public checks recompute statistics from the released aggregate strategy
paths and randomization distributions, and check stored aggregate estimates
and hashes. `scripts/reproduce_public_inputs.py` additionally refits the heads
and recomputes ranking and allocation from public sufficient statistics.
Transcript-to-feature reconstruction requires the conforming licensed inputs
described in `data/README.md`.

The separate `diagnostics/unsigned_displacement_20261011/` addendum contains
sector-level model scores and targets, aggregate monthly ICs, paired statistics,
and calendar-shift distributions for a known-period post-review diagnostic.
These are derived research outputs, subject to the same underlying market-data
boundary. Its replay recalculates statistics from the released scores; it does
not refit the unsigned heads or reconstruct licensed firm-level features.
The original six-head sufficient-statistic refit workflow is unchanged.
