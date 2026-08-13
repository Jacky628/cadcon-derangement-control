# Public anchor — repaired-extractor re-run (independent pre-registration)

This is the only pre-registration in this repository whose experiment is **still running** at
the time of its deposit. It is deposited now, before any result has been computed, so that the
decision rules and the reporting obligations below are on the public record ahead of the
numbers.

**The full document is deliberately not deposited yet, and the reason matters.** Its §1 names
this repository by its owning account, which would defeat the anonymised serving arranged for
the duration of double-blind review. Editing that line would change the document's hash and so
destroy the very anchor being deposited. The document is therefore held back, its SHA-256 is
deposited here, and the file itself will be added verbatim once review concludes — at which
point any reader can verify it against the hash below. Nothing about the decision rules is
withheld in the meantime: they are stated in full on this page.

## The defect being addressed

The generation extractor truncates each generation at its **first** top-level `result =`
assignment, while the scoring target is φ applied to the **untruncated** ground-truth program.
A model that faithfully continues into a second modelling stage is therefore not credited for
it. Measured on the frozen 400-program sample:

| | |
|---|---|
| reference programs that are multi-stage (≥2 top-level `result =`) | **202 / 400 (50.5%)** |
| 40% prefixes that already contain a complete `result` block | **68 / 400 (17.0%)** |
| programs whose four-tag target changes under the same truncation | **79 / 400** (70 by losing a CIRCLE) |

The published paper discloses the asymmetry and the 79/400 figure, and reports an
exclusion-based sensitivity analysis. An exclusion is not a repair, and the published record
says so.

## Rule A (frozen; the only thing that changes)

Take the **last** top-level `result =` cut whose program executes to a valid solid; fall back
to the current first-block behaviour when no cut executes. Degenerate EOS-less loops
(`result = (result...extrude(0.000000))`) require no hand-written detector: they make the whole
program fail to execute, so execution skips them on its own.

Sample, checkpoints, generation configuration, metric definition, test specification, decision
thresholds and guardrail thresholds are carried over from the mother pre-registrations
unchanged. Generation applies no extractor at all; both rules are applied offline to the same
stored completions, so the scoring surface is the only variable.

## Pre-registered outcome mapping, and what each obliges

Primary comparison: wrong-header minus never-header baseline, at the **prefix-cluster** unit
(298 distinct model inputs), on the Rule A surface. Text and token tokenizations are judged
separately; the more conservative outcome governs.

| Outcome | Criterion | Obligation |
|---|---|---|
| **CONFIRMED** | direction on ≥2/3 seeds **and** p<0.05 on ≥2/3 | Report as holding on the repaired surface. arXiv update optional. **Strengthening the existing wording is not permitted** — passing releases the limiting obligation, it does not grant licence to claim more. |
| **QUALIFIED** | direction holds, significance does not | **Mandatory** arXiv revision limiting the strength of the published claim. |
| **OVERTURNED** | direction reverses, or reverses significantly | **Mandatory** arXiv revision explicitly limiting or withdrawing the corresponding claim. |

The verbatim wording for all three outcomes was written before launch and is fixed
(document §10). Results from this protocol **replace no published number**: this is an
independent protocol, reported alongside the existing one, and no figure from the repaired
surface may enter the existing paper's results (document §9.1).

A pre-declared known cost, recorded before any result: Rule A widens the meaning of
`executable` from "the first block executes" to "some cut executes", which may weaken the
paper's own non-selection argument (text wrong-header executability 0.638 > baseline 0.603).
Both definitions are recorded per row so that a change can be attributed to the definition
rather than to the data, and the attribution table is fixed in advance (document §7.1).

## Timing — stated plainly

* The pre-registration and all ten freeze artifacts were committed, and their SHA-256 hashes
  registered, **before the first generation job started** (2026-08-12 01:54:19 UTC). That
  ordering is evidenced by local version-control history only.
* **This public deposit was made at 2026-08-12 02:14:52 UTC — 20 minutes and 33 seconds
  after generation began** (commit `4cac027`; the figure is stated to the second because
  precision about ordering is the only thing this file exists to provide). No cell had completed and no score had been computed, but a reader
  should not take this deposit as externally verifiable evidence that the document predates
  the data. What it does establish is that these rules, thresholds and obligations were fixed
  before any result existed.
* This pre-registration is, in addition, **fully unblinded**: the effects it re-examines are
  already published. The document says so in its own §1. The protection against post-hoc
  choice is the frozen decision chain and the fixed outcome wording, not blindness.

## Freeze artifacts

SHA-256 for all ten are in `../replication_frozen/extractor_fix_sha256s.txt`. One revision was
made after freezing and before any data existed — a relative path that made every worker fail
before loading a model; it touched no criterion, threshold, guardrail, extraction rule or
outcome mapping. Both the original and revised hashes are listed, and the reason is recorded
in `../replication_frozen/REPLFIX_RUN_LOG.md`.

## Outcome (added 2026-08-13, after completion)

The run completed on 2026-08-12 (27/27 cells, no failures). The gate passed first: the legacy
rule applied to the new generations reproduced the published geometry verdicts on all
10,800 rows and the published p-values digit for digit, so the two surfaces differ only by the
extraction rule. The pre-registered outcome is **CONFIRMED** under both tokenizations: the
wrong-versus-never-conditioning deficit holds direction on 3/3 seeds at the cluster unit on
the repaired surface, and the text seed that fails the significance threshold in the published
protocol fails here as well — it was not rescued.

One guardrail failed, and it bounds what the confirmation is worth: on 62.7% of multi-stage
generations the execution-driven rule returns exactly what the first-block rule returns,
because the later modelling stages the model writes are themselves usually not executable.
The two surfaces differ on 11.2% of scored rows. Per the pre-registration this may not be
described as "the conclusion is unchanged after the fix"; the accurate statement is that the
deficit reproduces under the limited bite the repair achieves. The declared risk to the
paper's non-selection argument did not materialise: the executability lift is 0.0 pp in all
nine conditions, so both definitions of `executable` coincide on every row.

Scoring code, per-row scores on both surfaces, the machine verdict and the run log are in
`../extractor_fix/` and `../replication_frozen/REPLFIX_RUN_LOG.md`. No figure from this
protocol replaces any number in the released paper; it is an independent protocol reported
alongside. The pre-registration document itself remains withheld for the duration of
double-blind review (see above); its SHA-256 is on record.
