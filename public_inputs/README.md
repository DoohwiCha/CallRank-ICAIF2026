# Public fixed-head replay inputs

This directory supplies the numerical inputs used by
`scripts/reproduce_public_inputs.py`. The replay independently fits the
PCA-32/Ridge-10000 heads, scores evaluation months, and computes rank IC and
42-session allocation results from these inputs. It starts at the sector-level
sufficient-statistic boundary. Encoder, scaler-input, and firm-history
reconstruction require the licensed upstream data described in `data/README.md`.

The six `raw_128`, `raw_160`, `raw_200`, `unit_128`, `unit_160`, and `unit_200`
NPZ files each contain two float64 arrays: a 703-by-703 training Gram matrix
and a 265-by-703 evaluation-to-training cross-Gram matrix. These represent
training-only standardized and centered sector features. PCA scores are
recovered spectrally from their leading 32 components. The matrices contain
sector-month inner products; call text, firm vectors, and issuer identifiers
are excluded.

`train_keys.csv` and `eval_keys.csv` supply sector-month keys, entry/exit dates,
and complete 42-session sector-minus-SPY target returns. Training outcomes end
by December 1, 2020; evaluation signal months begin in November 2023.
The target percentile ranks are calculated anew by the replay.

`sector_total_return_indices.csv` contains normalized dividend-adjusted
SPY and eleven sector-ETF series. Each series starts at one on its first
finite observation. All returns and price-ratio signals retain their original
values under this rebasing. `daily_factors.csv` contains the public French
factor observations; `DGS3MO.csv` contains Treasury yields used for the
strictly-prior daily risk-free adjustment.

`manifest.json` records exact input hashes, byte counts, array shapes, CSV
schemas, source attribution, and the fixed research settings. The public
validator permits only these specified numerical files. Underlying third-party
market, factor, and yield inputs retain their original source rights and terms;
see `CODE_AND_DATA_NOTICE.md`.
