# Frozen identity-masking input contract

`scripts/identity_masking.py` is the frozen masking engine used for the
reported full-pipeline identity-control experiment. It contains no transcript,
issuer, return, price, or provider row. Reproducing the transformed corpus
requires the licensed identity inputs below; those rows are intentionally not
redistributed.

The input Q&A must be the canonical clean-room `qa_doc` bytes after the study's
9,000-character cap. The masker receives those capped bytes directly; it does
not run a separate pre-mask whitespace-normalization pass. Whitespace is
normalized implicitly when the resulting masked document is split by words and
joined into non-overlapping 128-, 160-, and 200-word passages. The same
operation must be applied to fitting and evaluation calls.

The call-section table must preserve these frozen fields:

- `issuer_call_id`, the stable clean-room call key;
- canonical, original, and source company labels;
- canonical, original, source, and query ticker/symbol labels;
- `opening_claimed_issuer` from the transcript opening;
- `prepared_doc`, used only to recover prepared-section speaker headers; and
- the canonical, capped `qa_doc` that is transformed.

Two additional return-blind registries are required:

- a dated issuer metadata table with historical company names and ticker
  symbols, used to construct corpus-wide aliases; and
- a Q&A participant table keyed by `issuer_call_id`, with
  `question_speaker` and pipe-delimited `answer_speakers` fields.

`build_global_aliases`, `load_pair_rosters`, and `build_call_identities` turn
those inputs into the exact alias objects consumed by `IdentityMasker`. The
engine gives deterministic call-local aliases priority over broad named-entity
spans, applies context-only rules to one- and two-character tickers, and fails
the audit if a deterministic alias remains after masking. Broad named entities
use spaCy 3.8.14 with `en_core_web_sm` version 3.8.0 and the `PERSON`, `ORG`,
`PRODUCT`, `NORP`, and `FAC` labels.

The source module is sufficient to reproduce the masking transformation only
when these canonical rows, historical aliases, participant rosters, text
bytes, and pinned spaCy model are identical. A reduced user-created alias list
can run the public workflow, but it is not claimed to recreate the manuscript's
masked text byte for byte. The released monthly IC and circular-shift files are
the exact offline replay of the reported aggregate experiment.
