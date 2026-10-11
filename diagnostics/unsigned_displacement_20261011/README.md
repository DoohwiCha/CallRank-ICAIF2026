# Unsigned-displacement diagnostic

This addendum evaluates the element-wise absolute value of firm-conditioned
displacement on the already observed 24-month evaluation sample. It is a
post-review diagnostic, with unadjusted p-values, rather than a new holdout or
independent confirmation. Original signed results and public sufficient-statistic
inputs remain unchanged.

For each of the 128-, 160-, and 200-word passage lengths, the private execution
first constructs the same strictly prior expanding firm reference and signed
displacement `d`. It averages one of `d`, `d / ||d||`, `abs(d)`, or
`abs(d / ||d||)` across firms within each sector. Absolute value is applied
**before** sector averaging in the fixed encoder coordinate basis. Signed and
unsigned arms all use training-only scaling, full-solver PCA with 32 components,
and Ridge with penalty 10,000. A separate raw/PCA1 arm is retained as a capacity
diagnostic. Training contains 703 sector-month rows with outcomes completed by
December 1, 2020. The 265 scored rows contain 24 eligible months with six or more
sectors. Within-month percentile ranks are averaged across passage models.

`protocol.json` records the locally frozen specification, source input hashes
and the source execution protocol hash. Original raw/unit ensemble scores were
reproduced exactly before interpreting new results. Local freezing before
execution is not an external preregistration. The source call vectors are
licensed and are not included.

| File | Content |
| --- | --- |
| `predictions_by_chunk.csv` | 3,975 sector-level scores, five arms and three lengths |
| `ensemble_predictions.csv` | 1,325 sector-level rank-averaged scores |
| `monthly_ics.csv` | 24 monthly ICs for each of five arms |
| `rank_summary.csv` | Per-length and ensemble IC, NW(1), NW(2), and ensemble shift statistics |
| `paired_ic_tests.csv` | Four frozen contrasts; NW(1), NW(2), confidence intervals, and conventional paired t p-values |
| `angular_monthly_ic.csv` | Existing angular-control monthly ICs for paired comparisons |
| `shift_null_*.csv` | All 20,000 calendar-shift values for each arm |
| `fit_audit.csv` | Aggregate sample and feature-dimension checkpoints |
| `run_summary.json` | Original diagnostic execution and exact baseline reproduction receipt |
| `manifest.json` | File hashes, byte counts, data scope and replay boundary |

Run from the repository root:

```bash
python scripts/recompute_unsigned_diagnostic.py
```

The replay reconstructs rank ensembles, all monthly ICs, NW(1)/NW(2) inference,
paired t tests, and all 100,000 calendar-shift values from the released
sector-level scores. It checks the stored results and file hashes. It **does not
refit models**: the unsigned arms do not have public training Gram inputs.
The existing `scripts/reproduce_public_inputs.py` still refits six signed
raw/unit heads from its separately scoped public sufficient statistics.

Read score CSVs with `float_precision="round_trip"` in pandas. Exact ensemble
score ties affect Spearman ranks and shifts; default parsing can perturb ties.
The shift null preserves sector identity and persistent score differences, so
its mean need not be zero. NW tests mean IC against zero; these are distinct nulls.

The raw-minus-unsigned mean IC gap is .180580 (NW(1) p=.032171;
NW(2) p=.038033; paired t p=.065442). The unit-minus-unsigned gap is .069661
(NW(1) p=.526606; NW(2) p=.488763; paired t p=.525241).
The unit comparison does not demonstrate a signed advantage. All results,
including the weak one-PC result and conventional t sensitivity, are retained.
No portfolio experiment is added by this diagnostic.
