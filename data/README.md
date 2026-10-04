# Licensed input contract

No licensed or provider-derived row is stored in this directory. This document
describes upstream transcript-to-feature reconstruction, which reads licensed
or locally derived files from `--data-dir`. The separate `public_inputs/`
directory provides sector-level sufficient statistics and derived market inputs
for offline head, ranking, and portfolio replay without those upstream files.

The one public empirical input in this directory is
`DGS3MO_2023-10-01_2026-06-05.csv`, downloaded from the Federal Reserve Bank
of St. Louis FRED series DGS3MO. Sharpe ratios map each strategy return date
to the latest strictly prior observation and convert the annual percentage
yield to a compounded daily return. This macro series contains no licensed
transcript, membership, capitalization, provider-price, prediction, or
holding row.

## Clean-room event map

`scripts/validate_canonical_ledger.py` accepts the return-blind event map used
to create the call-section table:

| Column | Meaning |
|---|---|
| `issuer_call_id` | Stable call identifier |
| `canonical_issuer_id` | Issuer resolved from ticker, opening identity, executives, and corporate history |
| `verified_call_date` | Actual call date or externally confirmed date within the call month |
| `canonical_sector` | Sector effective on the verified call date |
| `cleanroom_usable` | Final inclusion flag |
| `content_sha256` | Licensed source-text hash |
| `canonical_pit_membership_verified` | Point-in-time membership check |
| `fiscal_year`, `period` | Reported fiscal period |

The validator checks the reported hash for the clean-room snapshot, row and issuer
counts, missing dates, duplicate firm events, cross-issuer content collisions,
fiscal-period validity, and point-in-time membership.

## Canonical call-section input

`scripts/build_embeddings.py` accepts one canonical call-section table:

| Column | Meaning |
|---|---|
| `security_id` | Stable issuer identifier |
| `ticker` | Ticker effective on the call date |
| `sector` | Sector label effective on the call date |
| `earnings_date` | Verified actual call date used to assign the calendar month |
| `membership_date` | Latest membership observation used for the call |
| `in_index` | Point-in-time S&P 500 membership flag |
| `prepared_text` | Licensed prepared-remarks text |
| `qa_text` | Licensed Q&A-side text |

The table is the output of the return-blind clean-room event map. Company
investor-relations records and dated webcast pages take precedence for call
dates; SEC filings and independent schedules provide supporting evidence.
Earnings-release and filing dates alone never move a call across calendar
months. The ledger resolves issuer mismatches, removes duplicate
issuer/content and issuer/event rows, and attaches the dated sector label.
The final return-blind rebuild restores 34 verified calls from one omitted
issuer and removes two wrong-issuer training calls. Aggregate counts and hashes
are released in `aggregate_results/canonical_data_summary.json` and
`aggregate_results/cleanroom_integrity_summary.json`.

The embedding builder rejects membership observations dated after a call and
rows marked outside the index. It applies minimum length filters, retains the
first 7,000 prepared-section and 9,000 Q&A-side characters, and then
normalizes whitespace. Each section is split into non-overlapping 128-, 160-,
and 200-word passages; the final partial passage is retained. Passage
embeddings are averaged within the call and normalized.

## Identity-control inputs

### Frozen inputs for the reported experiment

The reported identity controls use
`scripts/build_identity_controls.py --frozen-inputs`. The clean-room
call-section table must contain the exact
already capped Q&A bytes and stable keys below; the builder does not derive a
new call identifier or run the ordinary whitespace-cleaning/capping stage
again.

| Column | Meaning |
|---|---|
| `issuer_call_id` | Stable clean-room call identifier preserved in every arm |
| `asset_symbol`, `query_symbol`, `company` | Source ticker and company labels used by the frozen alias registry |
| `canonical_company` | Adjudicated company label used in the matched identity shell |
| `canonical_asset_symbol`, `canonical_query_symbol` | Adjudicated point-in-time ticker labels |
| `canonical_issuer_id` | Stable issuer identifier used for prior-firm history |
| `canonical_sector` | Point-in-time sector label |
| `original_asset_symbol`, `original_query_symbol` | Historical source-symbol aliases |
| `verified_call_date`, `verified_month` | Adjudicated call date and corresponding calendar month |
| `prepared_doc` | Capped prepared section used only to recover strict line-header speaker names |
| `qa_doc` | Canonical Q&A bytes after the 9,000-character cap and before identity masking |
| `opening_claimed_issuer` | Return-blind issuer phrase parsed from the transcript opening |
| `qa_capped_word_count`, `qa_text_sha256` | Whitespace word count and SHA-256 of the supplied `qa_doc` bytes |
| `cleanroom_usable` | Final return-blind inclusion flag |

The participant-pair table supplies current-call speaker order:

| Column | Meaning |
|---|---|
| `issuer_call_id` | Stable clean-room call identifier |
| `pair_index` | Within-call Q&A pair order |
| `question_speaker` | Parsed question speaker |
| `answer_speakers` | Pipe-delimited answer speakers in observed order |

The historical issuer-metadata table supplies `Name` and `Symbol` columns.
These rows expand the corpus-wide company and ticker registry beyond the
current call labels. The exact alias rules, priorities, short-ticker safeguard,
case-sensitive single-word issuer rule, spaCy version, placeholders, and
frozen aggregate checkpoints are documented in
`IDENTITY_MASKING_PROTOCOL.md` and
`../config/identity_control_protocol.json`.

The full-mask arm embeds the scrubbed capped Q&A. For the matched-body control,
the builder creates a fixed identity shell from canonical company, canonical
ticker, and speakers in first-occurrence order, then appends either the
target's scrubbed body or a one-to-one same-month cross-issuer/cross-sector
donor body. Donors minimize absolute log1p distance in the original capped-Q&A
whitespace word count with the frozen SHA-256 tie breaker. The composite is
not capped again. A fixed prepared-section scaffold is used for the two body
arms because the reported identity table evaluates the Q&A raw-displacement
and unit-direction representations, not a prepared-section representation.

The donor map, transformed text, aliases, embeddings, and row-level
predictions remain licensed and are not distributed. The public release
contains the exact 24 monthly ICs, 20,000-draw masked timing nulls, and
aggregate test summaries.

### Portable reduced-alias reference

Without `--frozen-inputs`, the same builder accepts the ordinary canonical
section schema plus `canonical_company`, `canonical_query_symbol`,
`company_aliases`, `ticker_aliases`, and ordered `speaker_aliases` cells, or a
long `--aliases` table with `issuer_call_id`, `alias_type`, and `alias`.
It applies the ordinary minimum-length filter and character cap before the
reduced alias mask. This path is useful for adapting another licensed source,
but it is not claimed to recreate the manuscript's transformed corpus byte for
byte.

## Derived model inputs

| File | Required columns | Meaning |
|---|---|---|
| `call_prepared_all128.parquet` | call keys, `e000` ... `e767` | Prepared-section call embeddings |
| `call_qa_all128.parquet` | same keys and coordinates | Q&A-side call embeddings |
| `call_prepared_all160.parquet` | same schema | 160-word prepared embeddings |
| `call_qa_all160.parquet` | same schema | 160-word Q&A embeddings |
| `call_prepared_all200.parquet` | same schema | 200-word prepared embeddings |
| `call_qa_all200.parquet` | same schema | 200-word Q&A embeddings |
| `targets.csv` | `sector`, `month`, `entry_date`, `exit_date`, `horizon`, `excess_return` | 42-trading-day sector ETF return minus SPY |
| `sector_etf_adjusted_close.csv` | `Date`, `SPY`, eleven sector ETF columns | Yahoo Finance dividend-adjusted closes for targets and daily paths |
| `pit_sector_market_caps.csv` | `Date`, `sector`, `market_cap` | Active-constituent sector capitalization at each close |

The call embedding key is `issuer_call_id`, `issuer_id`, `sector`, and
`month`. Prepared and Q&A files must have identical keys. `issuer_id` remains
stable across ticker changes. Sector labels may change over time.

## Reported-earnings input

The identification runner accepts a dated reported-earnings table with
`ticker`, `date`, `epsActual`, `epsEstimated`, `revenueActual`, and
`revenueEstimated`. The manuscript's point-in-time control uses only current
and four-quarter-lagged actual EPS and revenue. Estimate-based surprise rows
remain diagnostic because the licensed snapshot does not establish the
historical vintage of every analyst estimate.

## Point-in-time rules

- The historical firm reference uses only earlier firm-month Q&A embeddings.
- At least two prior firm-months are required before a residual is formed.
- Firms are equal-weighted within each sector-month.
- Training includes only targets whose 42-day outcome is complete by
  December 1, 2020.
- Evaluation starts with November 2023 signal months.
- Sector capitalization is observed at the trading close before entry.
- ETF returns start on the first trading day after month-end aggregation.
- On the first common portfolio date, both strategy and SPY returns start at
  zero; strategy entry costs are charged on that date. Subsequent returns use
  the same close-to-close interval for every path.

Historical membership comes from a licensed point-in-time constituent table.
Sector labels are attached at the verified call date. The mapping incorporates
the major 2016, 2018, and 2023 taxonomy changes without using future returns.
Fully dated firm-specific classification histories remain an external-input
extension rather than an inferred field in the public code.

The training-ledger manual audit uses a deterministic, return-blind hash
order, samples up to two calls per sector-year while prioritizing distinct
issuers, and adjudicates 149 calls against external issuer and date evidence.

## Provider boundary

Provider acquisition and credentials are outside this repository. Users must
supply their own licensed transcript, membership, capitalization, and price
files. No script sends a provider request or prints a credential.

The aggregate release also includes the public market, size, value,
profitability, investment, momentum, and risk-free factors from the Kenneth
French Data Library, restricted to the manuscript evaluation dates. These
series support offline recomputation of the reported CAPM and FF5+Momentum
alphas.
