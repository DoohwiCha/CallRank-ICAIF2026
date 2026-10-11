# CallRank

Code and aggregate results for **CallRank: Isolating What Changed in
Earnings-Call Q&A for Sector Ranking and Cost-Adjusted Alpha**, accepted at
ICAIF 2026 (paper 682).

Authors, in the CMT registration order: **Doohwi Cha, Minsuk Sung, Seunghan Son,
Juyeong Lee, Donghwa Seo, Minjae Lee, Hyeonjun Yeo, and Yoontae Hwang**.

Public repository: <https://github.com/DoohwiCha/CallRank-ICAIF2026>.

Current version: the October 11, 2026 reviewer revision, before the
corresponding author finalizes the manuscript in Overleaf. It includes the
unsigned diagnostic below and retains the accepted-snapshot numerical outputs.
The snapshot hashes are recorded in `PUBLIC_RELEASE.json`; later manuscript
changes will be synchronized separately. ACM/TAPS production is pending.

This repository contains the public implementation, fixed configuration,
aggregate results, public sufficient-statistic replay inputs, input contracts,
and synthetic tests. It is a named
camera-ready release of the accepted manuscript's current review artifact,
whose reported experiment uses 24 eligible post-release signal decisions.
The original frozen numerical configuration, empirical tables, strategy paths,
and figures are retained. A separate post-review diagnostic addendum releases
unsigned-displacement comparisons. Private development history and
raw calls and licensed firm-level input records are excluded.

The package contains no credential, raw transcript, call or firm record,
firm-level embedding, issuer identifier, historical membership row,
capitalization row, or actual account holding. Model selections and allocations
can be recomputed from the public sector-level sufficient statistics and
normalized market indices. Original provider dollar-price quotes are excluded.

## Reported design

For each eligible firm-month, the code:

1. embeds non-overlapping Q&A-side passages with the pinned SEC-domain sentence
   encoder;
2. subtracts the expanding mean of the same firm's earlier Q&A embeddings;
3. equal-weights either the raw signed residual or its unit direction across
   firms within a sector;
4. fits a separate PCA-32/Ridge head for every signed form and passage length
   using outcomes completed by December 1, 2020, then leaves those coefficients
   unchanged during evaluation; and
5. averages within-month prediction ranks from 128-, 160-, and 200-word
   passage models.

Raw displacement tests the complete signed movement in the full sector order.
Unit direction bounds each issuer's contribution before sector aggregation and
supplies the reported allocation. Their paired IC difference is not
statistically significant, so the artifact does not treat either signed form
as empirically dominant. Scalar angular distance is the movement-size control;
the separate diagnostic below also compares signed and element-wise unsigned
vectors with the same 32-PC head.

The reported allocation maps each monthly unit-direction ranking to one
argmax sector. Each monthly Top 1 sleeve remains open for 42 trading days.
Successive sleeves overlap, so as many as three monthly sleeves—and, when their
leaders differ, three distinct sector ETFs—can be active at once. Capital is
divided equally across active sleeves. Bottom 1 and the 100%-long/100%-short
Top 1-minus-Bottom 1 spread are reported diagnostics. All paths apply a
five-basis-point one-way trading cost. The spread deducts ETF turnover costs
but does not model borrow fees or financing charges.

## Identity-controlled mechanism tests

The release contains the exact aggregate replay and the frozen text
transformation used for the two controls reported in the manuscript:

- **Full-pipeline identity mask.** Company names, tickers, speaker names, and
  broad named-entity spans are replaced in the Q&A text before embeddings are
  rebuilt. Training and evaluation are both transformed, and both signed heads
  are refit using the same pre-2021 outcome boundary.
- **Matched identity-shell body swap.** Each target call receives the same
  company, ticker, and current-speaker shell in both arms. One arm appends the
  target's masked Q&A body; the other appends a deterministic, return-blind,
  same-month cross-sector donor body. The paired monthly IC difference isolates
  the incremental information in the correctly matched body conditional on the
  common shell.

With `--frozen-inputs`, `scripts/build_identity_controls.py` calls the frozen
masking engine in `scripts/identity_masking.py` and constructs all three
licensed-text variants from the canonical capped Q&A bytes, historical issuer
metadata, and participant-pair roster. `scripts/compare_identity_controls.py`
then compares completed fixed-head evaluations. The exact licensed schemas and
transformation boundary are documented in `data/README.md`,
`data/IDENTITY_MASKING_PROTOCOL.md`, and
`config/identity_control_protocol.json`.

The same builder also has a reduced-alias path for users whose licensed source
has only company, ticker, and speaker alias cells. That path is a portable
reference implementation; it is not claimed to recreate the manuscript's
masked corpus byte for byte. The released 24-month ICs and 20,000-draw nulls
are the exact offline replay of the reported identity-control results.

## Post-review unsigned-displacement diagnostic

[`diagnostics/unsigned_displacement_20261011/`](diagnostics/unsigned_displacement_20261011/)
contains a diagnostic on the already observed evaluation period. Before sector
averaging, each firm's signed displacement `d` is replaced by `abs(d)`, or its
unit direction `u` by `abs(u)`, element by element in the fixed encoder basis.
Both signed/unsigned comparisons retain the same 32-PC/Ridge head specification.
The addendum also retains a one-PC raw-displacement diagnostic.

| Ensemble | IC | NW(1) p | NW(2) p | Calendar-shift p |
| --- | ---: | ---: | ---: | ---: |
| Signed raw | .158 | .007 | .002 | .044 |
| Unsigned raw | -.023 | .795 | .794 | .877 |
| Signed unit | .100 | .055 | .030 | .099 |
| Unsigned unit | .031 | .711 | .686 | .389 |
| Signed raw, one PC | .017 | .750 | .698 | .430 |

The raw-minus-unsigned paired gap is .181 (NW(1) p=.032; NW(2) p=.038;
conventional paired t p=.065). The unit-minus-unsigned gap is .070
(NW(1) p=.527; NW(2) p=.489; paired t p=.525). All p-values are unadjusted.
This is a post-review diagnostic with 24 eligible months, not a new holdout or
independent confirmation. The protocol was frozen locally before these new
fits; this does not constitute an external preregistration.

The public addendum supplies sector-level passage-model scores, ensemble scores,
monthly ICs, all five 20,000-path shift distributions, and paired results.
Its replay **recomputes statistics from scores**; it does not refit unsigned heads
or reconstruct firm features. The original six-head raw/unit sufficient-statistic
refit remains a separate workflow. No transcript or firm embedding is added.

```bash
python scripts/recompute_unsigned_diagnostic.py
```

## Repository layout

```text
aggregate_results/       Aggregate tables, daily strategy paths, and figures
config/                  Frozen model, evaluation, and identity-test settings
diagnostics/              Separately scoped post-review score replay
data/README.md           Licensed-input schemas and clean-room boundary
public_inputs/           Sector-level sufficient statistics and replay market inputs
scripts/                 Feature, target, model, portfolio, and audit code
tests/                   Synthetic leakage, masking, and accounting tests
```

The aggregate directory records 21,421 final clean calls, including 4,604 in
the 32-month post-release census. The evaluation funnel contains 29 months with
complete targets and 24 months meeting the six-sector decision rule. It also
contains 20,000 temporal draws, 2,000 random-sector paths, 1,000 same-sector
wrong-firm draws, public factor series, aggregate unit/raw Top 1, Bottom 1, and
spread paths, and all nine raw-displacement robustness estimates. No row in the
release identifies a firm or an actual account holding.

## Manuscript-facing checkpoints

| Evidence | Reported result | Released source |
| --- | --- | --- |
| Full-sector ordering | Raw IC .158 (NW/shift p .007/.044); unit IC .100 (.055/.099); paired difference p .200 | `representation_results.csv`, `circular_shift_tests.csv`, `paired_ic_tests.csv` |
| Identity mask | Raw IC .191 (.002/.006); unit IC .183 (<.001/.004) | `identity_control_tests.csv` |
| Matched body swap | Correct-body minus donor-body gap .195 (p .020) for raw and .177 (p .027) for unit | `identity_control_tests.csv` |
| Unit-direction Top 1 | 42.1% annual net return versus 24.0% for SPY; 27.0% SPY alpha (NW(5)/NW(42) p .003/.004); 9.9% maximum drawdown | `portfolio_summary.csv`, `alpha_lag_sensitivity.csv` |
| Overlapping implementation | Up to three monthly Top 1 sleeves are active concurrently and capital is equal-weighted across them | `portfolio_daily_returns.csv`, `config/frozen_config.json` |
| Cost sensitivity | At 20 bp one way: 39.6% annual return, 24.8% SPY alpha (p .007/.008), and 10.7% maximum drawdown | `cost_sensitivity.csv` |
| Firm matching | Own-history direction IC .135 (NW p .013); same-sector wrong-firm p .005; issuer retrieval 41.4% versus 3.2% | `history_reference_tests.csv`, `wrong_firm_placebo_draws.csv`, `issuer_identity_probe.csv` |
| Raw robustness | All nine estimates are positive; six 95% NW intervals exclude zero; reported-earnings partial IC .171; history-at-least-eight IC .144 | `raw_robustness.csv`, `figures/callrank_robustness.pdf` |
| Sector concentration | The leader spans ten sectors; static XLU returns 17.4% annually versus 24.0% for SPY | `top1_selection_frequency.csv`, `static_sector_benchmark.csv` |

## Validate without licensed data

Python 3.12 is recommended. `requirements-validation.txt` records the exact
environment used for the published package checks and figure regeneration.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-validation.txt
python scripts/reproduce_public_inputs.py
python scripts/build_manuscript_figures.py
python scripts/recompute_published_results.py
python scripts/recompute_unsigned_diagnostic.py
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

These checks require no API key, proprietary file, or network access after
dependencies are installed. `reproduce_public_inputs.py` estimates the
PCA-32/Ridge heads from public train-only standardized Gram matrices, then
computes new evaluation scores, monthly rank ICs, sector selections, overlapping
42-session portfolio returns, and alpha. These sufficient statistics preserve
the fixed-head calculation without releasing call text or firm vectors. This
replay starts at the sector-level sufficient-statistic boundary; rebuilding the
encoder and upstream firm-conditioned features requires licensed inputs.

The remaining checks recompute the released risk and alpha
statistics, long-lag inference, cost sensitivities, randomization p-values,
and identity-control checkpoints. They check the stored raw-robustness
estimates, three-sleeve accounting rule, and every aggregate-file hash. Raw
robustness re-estimation and upstream text/embedding reconstruction require
the licensed inputs. The identity-control aggregates have their own release
scope in `config/identity_control_protocol.json`; their firm embeddings remain
excluded. The checks also scan the distributable tree and reachable
Git history for credentials, local paths, unsupported version claims, and
excluded row-level data; named paper authors and public repository attribution
are permitted in this release.

## Licensed-input workflow

The exact schemas are documented in [`data/README.md`](data/README.md). A
licensed user first builds the ordinary CallRank inputs:

```bash
python scripts/validate_canonical_ledger.py \
  --ledger /licensed/cleanroom_event_map.parquet
python -m pip install -r requirements-embedding.txt
python scripts/build_embeddings.py \
  --input /licensed/cleanroom_call_sections.parquet \
  --output-dir /private/work/original
python scripts/build_targets.py \
  --prices /licensed/sector_etf_adjusted_close.csv \
  --months /private/work/original/call_qa_all160.parquet \
  --output /private/work/original/targets.csv
cp /licensed/sector_etf_adjusted_close.csv /private/work/original/sector_etf_adjusted_close.csv
cp /licensed/pit_sector_market_caps.csv /private/work/original/pit_sector_market_caps.csv
python scripts/run_all.py \
  --data-dir /private/work/original \
  --results-dir /private/work/original-results
```

To rebuild the reported identity controls, install the pinned spaCy model,
construct the three transformed section tables from the frozen licensed-input
contract, embed each table without reapplying the ordinary character cap, and
evaluate each embedding set with the same targets, prices, capitalization, and
fixed-head configuration:

```bash
python scripts/build_identity_controls.py \
  --frozen-inputs \
  --sections /licensed/identity_control_call_sections.parquet \
  --pair-rosters /licensed/conditional_qa_pairs.parquet \
  --historical-metadata /licensed/dated_issuer_metadata.parquet \
  --output-dir /private/work/identity-sections \
  --spacy-model en_core_web_sm \
  --seed 20260804

for arm in full_identity_mask correct_body donor_body; do
  python scripts/build_embeddings.py \
    --input /private/work/identity-sections/${arm}.parquet \
    --output-dir /private/work/${arm} \
    --preprocessed-identity-control
  cp /private/work/original/targets.csv /private/work/${arm}/targets.csv
  cp /licensed/sector_etf_adjusted_close.csv /private/work/${arm}/sector_etf_adjusted_close.csv
  cp /licensed/pit_sector_market_caps.csv /private/work/${arm}/pit_sector_market_caps.csv
  python scripts/run_core.py \
    --data-dir /private/work/${arm} \
    --results-dir /private/work/${arm}-results \
    --randomizations 20000 \
    --random-seed 20260804
done

python scripts/compare_identity_controls.py \
  --masked-results /private/work/full_identity_mask-results \
  --correct-results /private/work/correct_body-results \
  --donor-results /private/work/donor_body-results \
  --output /private/work/identity_control_tests.csv
```

The identity transformation never reads returns, targets, prices, or market
capitalizations. It verifies the original capped-Q&A hashes and word counts,
the frozen alias-registry checkpoints, residual-alias coverage, one-to-one
cross-issuer/cross-sector donor assignment, and identical paired identity
shells. The model runner separately enforces the pre-2021 fitting boundary,
target-key alignment, prior-only firm history, capitalization timing, and the
24-month coverage rule.

## License boundary

The MIT license applies only to source code in this repository. It does not
grant redistribution rights for transcripts, membership, capitalization,
prices, or derivatives at the call, firm, sector-month prediction, or holding
level. Aggregate result tables are included so reviewers can verify every
reported value without receiving restricted records.

This repository is the named public code release cited by the camera-ready
manuscript. The original anonymous review artifact is preserved separately.
