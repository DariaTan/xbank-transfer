# Cross-Institution Transfer of Event-Sequence Foundation Models for Banking Transactions

## 1. Task

Pretrain several self-supervised foundation models (FMs) on **MBD** (Sber
AI Lab's public *Multimodal Banking Dataset*, arXiv:2409.17587 — corporate/
legal-entity clients), then measure how well their **frozen** embeddings
transfer — with no re-training or fine-tuning of the FM itself — to
**xbank** (VTB, a second institution's private transaction data, also
companies/legal entities, identified by INN). Transfer quality is scored by
how well a small downstream probe, trained only on frozen embeddings,
predicts each dataset's own binary product-propensity targets.

Two things differ between the datasets, and this design treats them as
**two independent axes** rather than one bundled "new bank" shift:
- **Institution**: xbank vs. MBD — different bank, different client
  population, no shared category vocabulary (§4).
- **Aggregation level**: xbank's transaction table is **pre-aggregated to
  one row per (client, day, category-combination)**, collapsing however
  many raw transactions a client made in that day+category into one summed
  row; MBD keeps **raw, individual transactions at hour-of-day precision**,
  and is adapted (`data/mbd_adapter.py`) into BOTH forms — its native raw
  form, completely untouched (NOT an "hourly aggregation" — nothing is
  binned to an hour, each row is just one unmodified transaction), and a
  daily-aggregated version matching xbank's own row convention. The final
  deadline-constrained protocol pretrains once on MBD-raw and applies that
  same frozen checkpoint to all three evaluation corpora (§4); there is no
  MBD-daily pretraining arm.

In short: **does a transaction-sequence FM's representation generalize
across a bank boundary**, independently of whether it also has to
generalize across a shift from raw-event to daily-aggregated input — and
which of the five pretraining objectives degrades least on each axis?

> **Client segment note:** `configs/data/mbd.yaml`'s existing comment
> describes MBD as "corporate clients ... segment mismatch vs xbank's
> retail-consumer clients" — that comment is **stale/incorrect**: xbank's
> clients are companies (INN-identified legal entities), not retail
> individuals, same as MBD. There may still be a segment difference (e.g.
> SME vs. larger corporate, industry mix), but "retail vs. corporate" is
> not it — that yaml comment should be corrected separately, and any real
> segment-mismatch claim needs its own verification before being used as a
> novelty point.

## 2. Novelty

- **Five architectures, one protocol.** CoLES/NEP/MLM (embedding-style,
  self-supervised) and THP/COTIC (continuous-time point-process style) are
  usually evaluated separately in the literature. Here all five are
  pretrained on identical data/splits and probed identically, so differences
  are attributable to the objective, not to incidental protocol differences.
- **Institution shift and aggregation-level shift evaluated with one frozen
  source model.** A checkpoint pretrained once on MBD-raw is evaluated on
  MBD-raw, MBD-daily, and xbank-daily. MBD-raw vs. MBD-daily measures the
  within-institution aggregation shift; MBD-daily vs. xbank-daily measures
  the additional institution shift when both evaluation inputs are daily.
  This three-cell design does not identify a separate interaction term.
- **Aggregation-level mismatch as an explicit hypothesis, not a footnote —
  now in the direction of information LOSS, not novelty.** THP/COTIC's
  entire training signal is inter-arrival time between individual events.
  A model pretrained on MBD's raw, untouched, hour-precision data learns
  real sub-day dynamics; evaluated on xbank's already-daily-aggregated rows, that
  sub-day structure simply doesn't exist any more — every (client, day,
  category) is already one collapsed row, so the model sees, at best, one
  "event" per day per category rather than the many-events-per-day pattern
  it was pretrained to model. This predicts THP/COTIC will transfer
  *worse* under the aggregation shift than the non-temporal architectures
  (CoLES/NEP/MLM, which only lose count/amount granularity, not a whole
  modeling assumption) — a testable claim this design is built to produce
  evidence for, one way or the other.
- **Vocabulary-transfer failure mode reported, not hidden.** The category
  vocabularies a model actually learns during MBD pretraining (fit on
  whatever MBD field landed in each `col_2..col_16` slot, see §4) share no
  overlap with xbank's real category codes at eval time — a naive frozen-
  embedding transfer mostly loses categorical signal on the xbank side.
  Reporting *how much* of the transfer gap is attributable to this (vs.
  genuine representation mismatch) is itself a useful result for anyone
  reusing a bank-specific FM elsewhere.

## 3. Architectures

| Model | Family | Signal used | xbank checkpoint (historical reference baseline) | MBD checkpoint (primary, going forward) |
|---|---|---|---|---|
| CoLES | Contrastive metric learning (ptls) | Full event, subsequence splitter | Done, in-domain eval done | MBD-raw done |
| NEP | Autoregressive transformer (next-event prediction) | Full event | Done, in-domain eval partial | MBD-raw done |
| MLM | Bidirectional transformer, masked-event objective | Full event | Done, in-domain eval not started | MBD-raw done |
| THP | Transformer Hawkes Process (TPP, continuous-time intensity) | Event time + type only | Done, in-domain eval done | MBD-raw done |
| COTIC | Continuous-time convolutional TPP | Event time + type only | Done, checkpointed, inference not built | MBD-raw done |
| Chronos-2 | Zero-shot pretrained univariate time-series FM (external baseline, no xbank/MBD-specific training at all) | Amount series only (`col_11`) | Embeddings computed, in-domain eval done | N/A (zero-shot, no pretraining corpus at all) |

The five xbank checkpoints predate the 2026-09-14 direction decision (§1)
and are kept as a same-institution upper-bound reference — "what if the FM
had been pretrained on the eval institution itself" — not as an arm that
gets further developed. The production checkpoints are the five MBD-raw
checkpoints; MBD-daily is an inference corpus only.

Chronos-2 is a deliberate control: since it never trained on either bank's
data on any objective, its transfer "gap" (if any) isolates how much of the
other models' degradation is about *this* transfer, versus just the
difficulty of the downstream task itself.

## 4. MBD → xbank: Pretraining, Adaptation, and Transfer Design

MBD is already on disk (`/app/data/mbd/`), verified directly (not just from
the paper) on 2026-09-08:

- `detail/trx/fold={-1,0,1,2,3,4}` — transaction rows. `fold=-1` (562M rows)
  is the large **unlabeled** pool; folds 0–4 (~75–83M rows each) are the
  **labeled**, client-disjoint benchmark folds (`client_split/fold=k` lists
  ~200K client ids per fold, ≈1M labeled clients total, confirmed strictly
  disjoint by direct query on 2026-09-14 — 0 clients span more than one
  fold across `client_split`, `detail/trx`, and `targets`).
- Schema: `client_id, event_time (hour-precision, raw per-transaction rows),
  amount, event_type (56 vals), event_subtype (62 vals), currency
  (15 vals), src_type{11,12,21,22,31,32}, dst_type{11,12}, fold,
  is_balanced`. No column overlaps semantically with xbank's `col_2..col_16`
  beyond "amount" and "a category-like code" in the abstract.
- `targets/fold=k`: `client_id, mon` (12 monthly cuts, 2022-02 .. 2023-01),
  4 binary flags — `bcard_target` (0.44% positive), `cred_target` (0.05%),
  `zp_target` (0.38%), `acquiring_target` (0.26%). All four are rarer than
  any of xbank's four targets — expect undersampling / class-weighting to
  matter more here than for xbank's own targets.
- Modality: transactions only. MBD also has geo, support-dialogue
  embeddings, and product-purchase aggregates as separate modalities, all
  out of scope here — xbank's own pretraining corpus is transactions-only,
  so including MBD's other modalities would give the MBD-pretrained model
  signal xbank fundamentally cannot supply at eval time, confounding the
  institution-transfer question rather than isolating it. `data/
  mbd_adapter.py` already only ever touches MBD's transaction-level fields.

**Adaptation, now serving pretraining first (reversed from the original
eval-only design):**

1. `data/mbd_adapter.py` reshapes MBD's transactions into xbank's own
   `id, col_1, col_2..col_16` column convention, at either `freq=None`
   (**raw** — MBD's transactions completely untouched, one row per
   transaction, NOT an "hourly aggregation" — `configs/data/mbd.yaml`) or
   `freq="D"` (daily-aggregated to xbank's own (client, day, category) row
   convention — `configs/data/mbd_daily.yaml`). The raw adapted file is the
   actual pretraining corpus passed to `train_{model}.py`; the daily file is
   used only at inference time for the aggregation-shift measurement. Thus whatever
   category vocabulary a model's embedding tables learn comes from MBD's
   values sitting in xbank's named slots, not from xbank's own codes. Both
   adapted transaction variants include MBD's unlabeled `fold=-1` pool alongside the labeled
   folds (decided 2026-09-14, see point 4 below) — only the targets file
   stays restricted to labeled folds.
2. **No adapter is needed on the xbank side.** xbank's real transactions
   table is already shaped exactly like the adapter's output (`id, col_1,
   col_2..col_16`), which is the whole reason the adapter targets that
   shape — so zero-shot transfer eval is simply running an MBD-pretrained
   checkpoint through the existing `infer_{model}.py` path directly over
   xbank's own unmodified transactions/targets, once the checkpoint-source
   override described in §6 is built (today those scripts assume an
   xbank-pretrained checkpoint).
3. **One pretraining corpus, three evaluation corpora.** Every trainable
   architecture uses the same MBD-raw checkpoint:

   | # | Pretrain on | Eval on | Isolates |
   |---|---|---|---|
   | 1 | MBD-raw | MBD-raw | **in-domain reference** — no institution or aggregation change |
   | 2 | MBD-raw | MBD-daily | **aggregation shift** — same institution and labeled population, daily input at evaluation |
   | 3 | MBD-raw | xbank-daily | **total transfer shift** — institution and aggregation differ from pretraining |

   Comparing rows 2 and 3 estimates the additional institution effect
   conditional on both evaluation inputs being daily. With only these three
   cells, a separate aggregation-by-institution interaction is not
   identifiable and must not be reported as an independently measured term.

   Weekly aggregation was considered and dropped from this concrete design
   (not built, and a third aggregation level isn't needed to answer the
   institution-vs-aggregation question).
4. **Pretraining uses ALL folds, including the unlabeled pool (decided
   2026-09-14).** MBD's client-disjoint labeled folds (0-4) are what the
   in-domain MBD probe rotates through as train/test (§5), but pretraining
   itself is self-supervised and needs no labels — so it also draws on the
   large unlabeled `fold=-1` pool (562M rows), which MBD's own benchmark
   reserves for exactly this. `data/mbd_adapter.py`'s `ALL_FOLDS` (labeled
   + unlabeled) is now the default for the transactions file; the targets
   file stays restricted to the labeled folds regardless (fold=-1 has no
   targets partition at all). This substantially grows the pretraining
   corpus (raw: ~950M rows total vs. ~385M from labeled folds alone) —
   the already-materialized `mbd_data/daily_adapted/` build predates this
   decision and needs a rebuild (see §6).
5. **Vocabulary mismatch is a known, accepted limitation for v1.** A
   model's fitted category vocabularies come from MBD-adapted pretraining
   data; xbank's real category codes, passed through the same named slots
   at eval time, mostly resolve to an out-of-vocabulary/pad embedding.
   This means CoLES/NEP/MLM's categorical embedding tables contribute
   little on the xbank transfer eval — transfer there will lean almost
   entirely on amount and event-timing/count structure. **Report this
   explicitly** rather than let a transfer-gap number be misread as "the
   representation is bad" when it may just be "the category vocabulary
   doesn't transfer." A manual/learned category remapping (e.g.
   nearest-neighbor by frequency profile) is flagged as future work, not
   v1 scope.
6. **The column-slot assignment itself is arbitrary and UNTESTED — flagged
   2026-09-08, MORE consequential now than when it was only an eval-time
   choice.** `data/mbd_adapter.py` maps MBD's raw fields onto xbank's
   `col_2..col_16` NAMED SLOTS, but which MBD field lands in which slot
   was a plausible-sounding guess, not a verified choice — and now that
   this mapping decides what a model's embedding tables actually LEARN
   during pretraining (not just what gets fed through a frozen checkpoint
   at inference time), a bad assignment doesn't just add inference-time
   noise, it could shape the pretrained representation itself. Concretely:
   - **`amount → col_11`** (leaving `col_12`, xbank's other real numeric
     column, as a constant-zero placeholder) picks one of two
     *differently-initialized* numeric projections essentially by
     coin-flip, and that projection is then TRAINED on MBD's amount
     values specifically. Unlike the categorical case, there's no OOV
     buffer here — this is now a real modeling choice, not just a
     transfer-time routing decision.
   - **Categorical fields** (`event_subtype→col_3`, `src_type11→col_5`,
     etc.) were assigned "in whatever order MBD exposes them" — only
     `event_type→col_2` has any real rationale (both described as
     MCC-like). Each such slot's embedding table is now fit entirely on
     whichever MBD field was arbitrarily routed there.
   - **Action before trusting any MBD-pretrained transfer number**: run a
     slot-assignment ablation (e.g. remap `amount→col_12` instead of
     `col_11`, permute a couple of the arbitrary categorical mappings),
     re-pretrain, and check whether downstream probe metrics move
     meaningfully. If they do, that instability needs to be reported as a
     limitation, not absorbed into "the" transfer-gap number. Not yet run.

## 5. Key Technical Decisions

**ID / date splitting (MBD, in-domain / pretraining source):** client-level
folds are already disjoint by construction (verified above) — inherited
as-is rather than re-split. The paper (Sec 4.1) specifies an "out-of-fold
validation protocol" — 5 folds, 4 for training and 1 held out — but names
no canonical fold (confirmed by fetching the paper directly, 2026-09-08).
So the in-domain probe (`train_downstream_mbd.py`) rotates through every
fold as test in turn and reports results aggregated (mean ± std) across
all 5 rotations, not a single arbitrarily-picked fold's number.

**ID / date splitting (xbank, transfer target):** two independent leakage
guards, both already implemented and config-driven
(`configs/models/downstream.yaml`'s `probe:` section) — no longer called
"in-domain" now that no pretraining happens on xbank, but the same
calendar-based protocol still governs how the downstream probe is
trained/tested against xbank's own targets:
- *Row → client*: within the train period, train/val is split by unique
  client id (`GroupShuffleSplit`), never by row — a client's several
  monthly rows can't straddle train and val.
- *Time*: test is a held-out **calendar period** (train = all of 2023, test
  = 2024-01/02), not a random row split — the probe is never evaluated on a
  period it could have trained on. The config also warns (not errors) if a
  configured month has zero rows in the target file, since xbank's targets
  are known to skip Nov/Dec 2023 entirely.

This is a genuine asymmetry between the two datasets' own protocols
(fold-based for MBD, calendar-based for xbank) — inherited from each
dataset's own benchmark convention rather than forced into alignment, and
should be stated plainly in any writeup.

**Separate vs. joint target training:** independent per-target models
(one LightGBM + one MLP per target column, each with its own
`scale_pos_weight`/`pos_weight` correction and its own early-stopping
point) is the only arm now. A joint shared-trunk MLP (one model, 4 output
heads, trained once) was built and tested on CoLES specifically — it was
clearly *worse* on the common target (`col_2`: 0.801 vs. 0.821 AUC) and
only a coin-flip on the three rarer ones, so it was removed rather than
kept as a second comparison arm. Not re-tried on MBD before removal.

**Imbalance handling:** class weighting (`scale_pos_weight`/`pos_weight`,
always on, scaled from whatever training labels are actually passed in) and
an optional negative-undersampling cap (`max_neg_ratio` in
`configs/models/downstream.yaml`'s `probe:` section, per-target only, off by
default for xbank) are both available; the two aren't meant to be stacked
at full strength simultaneously (undersampling already softens the
automatically-computed class weight). A small grid (`max_neg_ratio` in
{5, 10, 20} on xbank/CoLES) showed the exact ratio barely matters once
undersampling is on at all — real jump vs. none, then flat — so the
committed default (20) was kept rather than tuned further. Likely worth
enabling for MBD's `cred_target` (0.05% positive) even if 20 stays the
default elsewhere.

## 6. Open Items Before Running the Full Comparison

- THP's downstream-probe MLP arm should get feature standardization before
  its first real run through either probe script — its embedding L2 norm
  (~76 median, after an architecture-level fix to an upstream `easy_tpp`
  bug) is still ~10-15x larger than CoLES/Chronos-2's, which could bias
  the MLP arm specifically (LightGBM is scale-invariant, unaffected). Not
  yet added.
- Run the MBD column-slot-assignment ablation described in §4 point 6
  before trusting any MBD-pretrained transfer number — swap `amount`'s
  slot (`col_11`↔`col_12`) and a couple of the arbitrary categorical
  mappings, re-pretrain, check whether downstream probe metrics move
  meaningfully. Not yet run.
- Build the checkpoint-source / eval-data decoupling described in §4 —
  `infer_mbd.py` currently hardcodes both which MBD variant it embeds
  (`configs/data/mbd.yaml`, always raw) and which checkpoint it loads
  (always the xbank-pretrained one, from `configs/models/<model>.yaml`'s
  `checkpoint_dir`); the xbank `infer_{model}.py` scripts likewise assume
  an xbank-pretrained checkpoint. None of §4's three experiments can run
  through the inference scripts as they stand today. Not yet started.
- Consider whether `infer_mbd.py`/`train_downstream_mbd.py` need to
  explicitly filter to labeled clients only when generating embeddings
  for the in-domain MBD probe — now that the transactions file includes
  `fold=-1`'s unlabeled clients too, naively sampling from "all clients in
  the transactions file" would waste compute embedding clients with no
  target to score against. Not yet checked.
- Pretrain all five architectures on MBD (both raw and daily) — the
  "MBD checkpoint" column in §3 is entirely not-started as of 2026-09-14.
