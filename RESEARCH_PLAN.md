# Cross-Institution Transfer of Event-Sequence Foundation Models for Banking Transactions

## 1. Task

Pretrain several self-supervised foundation models (FMs) on one bank's
client transaction stream (**xbank**, VTB — clients are companies/legal
entities, identified by INN, not individuals), then measure how well their
**frozen** embeddings transfer — with no re-training or fine-tuning of the
FM itself — to a second institution's data (**MBD**, Sber AI Lab's public
*Multimodal Banking Dataset*, ACM CIKM 2025, also corporate clients).
The two datasets differ in institution, and — importantly — in what a
"row" represents: xbank's transaction table is **pre-aggregated to one row
per (client, day, category-combination)**, collapsing however many raw
transactions a client made in that day+category into one summed row,
whereas MBD keeps **raw, individual transactions at hour-of-day
precision**. Transfer quality is scored by how well a small downstream
probe, trained only on frozen embeddings, predicts each dataset's own
binary product-propensity targets.

In short: **does a transaction-sequence FM's representation generalize
across a bank boundary**, and specifically across a shift from
daily-aggregated to raw-event input — and which of the five pretraining
objectives degrades least when it does?

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
- **Institution shift compounded by an aggregation-level shift, not just a
  new bank.** MBD isn't only "the same kind of data from someone else" — it
  swaps xbank's pre-aggregated daily summary rows (one row already sums an
  entire client-day-category's transactions) for raw, individual,
  hour-precision transactions. That's a harder and more realistic transfer
  test than a typical same-format cross-time or cross-account-type
  evaluation: the FM has to generalize to an input it was never structurally
  exposed to, not just a new distribution over the same input shape.
- **Aggregation/temporal-resolution mismatch as an explicit hypothesis, not
  a footnote.** THP/COTIC's entire training signal is inter-arrival time —
  pretrained on xbank, they never saw more than one event per client per
  category per day (multiple same-day transactions were already summed away
  before pretraining), so MBD's raw multi-event days and sub-day gaps are a
  structurally new input, not just a finer-grained version of the same
  thing. This predicts THP/COTIC will transfer *worse* than the
  non-temporal architectures (CoLES/NEP/MLM, which only lose some
  count/amount granularity, not a whole modeling assumption) — a testable
  claim this design is built to produce evidence for, one way or the other.
- **Vocabulary-transfer failure mode reported, not hidden.** MBD's
  categorical fields share no vocabulary with xbank's (see §4) — a naive
  frozen-embedding transfer mostly loses categorical signal on the target
  side. Reporting *how much* of the transfer gap is attributable to this
  (vs. genuine representation mismatch) is itself a useful result for
  anyone reusing a bank-specific FM elsewhere.

## 3. Architectures

| Model | Family | Signal used | Status |
|---|---|---|---|
| CoLES | Contrastive metric learning (ptls) | Full event, subsequence splitter | Pretrained on xbank, in-domain eval done |
| NEP | Autoregressive transformer (next-event prediction) | Full event | Pretrained on xbank, in-domain eval partial |
| MLM | Bidirectional transformer, masked-event objective | Full event | Pretrained on xbank, in-domain eval not started |
| THP | Transformer Hawkes Process (TPP, continuous-time intensity) | Event time + type only | Pretrained on xbank, in-domain eval done |
| COTIC | Continuous-time convolutional TPP | Event time + type only | Pretrained on xbank, checkpointed, inference not built |
| Chronos-2 | Zero-shot pretrained univariate time-series FM (external baseline, no xbank/MBD-specific training at all) | Amount series only (`col_11`) | Embeddings computed, in-domain eval done |

Chronos-2 is a deliberate control: since it never trained on either bank's
data on any objective, its transfer "gap" (if any) isolates how much of the
other models' degradation is about *this* transfer, versus just the
difficulty of the downstream task itself.

## 4. MBD Integration

MBD is already on disk (`/app/data/mbd/`), verified directly (not just from
the paper) on 2026-09-08:

- `detail/trx/fold={-1,0,1,2,3,4}` — transaction rows. `fold=-1` (562M rows)
  is the large **unlabeled** pool; folds 0–4 (~75–83M rows each) are the
  **labeled**, client-disjoint benchmark folds (`client_split/fold=k` lists
  ~200K client ids per fold, ≈1M labeled clients total).
- Schema: `client_id, event_time (hour-precision, raw per-transaction rows —
  unlike xbank's daily-aggregated ones, see §1), amount, event_type
  (56 vals), event_subtype (62 vals), currency (15 vals), src_type{11,12,21,
  22,31,32}, dst_type{11,12}, fold, is_balanced`. No column overlaps
  semantically with xbank's `col_2..col_16` beyond "amount" and "a
  category-like code" in the abstract.
- `targets/fold=k`: `client_id, mon` (12 monthly cuts, 2022-02 .. 2023-01),
  4 binary flags — `bcard_target` (0.44% positive), `cred_target` (0.05%),
  `zp_target` (0.38%), `acquiring_target` (0.26%). All four are rarer than
  any of xbank's four targets — expect undersampling / class-weighting to
  matter more here than in-domain.

**Integration plan:**

1. **No pretraining on MBD.** Every xbank-pretrained checkpoint is run
   through its existing `extract_embeddings`/inference path, unchanged,
   over MBD transactions — genuinely frozen, zero-shot.
2. **Reuse the existing windowing machinery** (`load_windowed_transactions_for_dates`)
   against MBD via a thin column-mapping adapter (`client_id→id`,
   `event_time→col_1`, `amount→col_11`, categorical fields left
   unmapped — see vocabulary point below), targeting MBD's own 12 monthly
   cuts rather than xbank's calendar grid.
3. **Respect MBD's own fold split for the probe**, not a re-derived one.
   The paper (Sec 4.1) specifies an "out-of-fold validation protocol" —
   5 folds, 4 for training and 1 held out — but names no canonical fold
   (confirmed by fetching the paper directly, 2026-09-08). So the probe
   rotates through every fold as test in turn and reports results
   aggregated (mean ± std) across all 5 rotations, not a single
   arbitrarily-picked fold's number. This is a *different* splitting axis
   than xbank's train/test-by-calendar-year (see §5) — the two protocols
   are not apples-to-apples beyond "did architecture ranking hold."
4. **Vocabulary mismatch is a known, accepted limitation for v1.** xbank's
   fitted category vocabularies (from `PandasDataPreprocessor`) have never
   seen MBD's category codes; passed through as-is, they mostly resolve to
   an out-of-vocabulary/pad embedding. This means CoLES/NEP/MLM's
   categorical embedding tables contribute little on MBD — transfer there
   will lean almost entirely on amount and event-timing/count structure.
   **Report this explicitly** rather than let a transfer-gap number be
   misread as "the representation is bad" when it may just be "the
   category vocabulary doesn't transfer." A manual/learned category
   remapping (e.g. nearest-neighbor by frequency profile) is flagged as
   future work, not v1 scope.
5. **The column-slot assignment itself is arbitrary and UNTESTED — flagged
   2026-09-08, not yet resolved.** `data/mbd_adapter.py` maps MBD's raw
   fields onto xbank's `col_2..col_16` NAMED SLOTS, but which MBD field
   lands in which slot was a plausible-sounding guess, not a verified
   choice, and the two aren't neutral containers: each xbank column has
   its OWN fitted, distinct weights from pretraining (a separate learned
   projection per numeric column, a separate embedding table per
   categorical column). Concretely:
   - **`amount → col_11`** (leaving `col_12`, xbank's other real numeric
     column, as a constant-zero placeholder) picks one of two
     *differently pretrained* numeric projections essentially by
     coin-flip. Unlike the categorical case, there's no OOV buffer here —
     a continuous value routed through `col_11`'s learned scale/weights
     vs. `col_12`'s would plausibly give a **systematically different**
     embedding, not just a noisier one.
   - **Categorical fields** (`event_subtype→col_3`, `src_type11→col_5`,
     etc.) were assigned "in whatever order MBD exposes them" — only
     `event_type→col_2` has any real rationale (both described as
     MCC-like). This is likely *partly* dampened by point 4's OOV
     argument (most values collapse to the same OOV/pad token regardless
     of slot), but not cleanly zero-impact: a low integer code could
     coincidentally match a different column's in-vocabulary index by
     chance, so the assignment still isn't provably neutral.
   - **Action before trusting any MBD transfer number**: run a
     slot-assignment ablation (e.g. remap `amount→col_12` instead of
     `col_11`, permute a couple of the arbitrary categorical mappings)
     and check whether downstream probe metrics move meaningfully. If
     they do, that instability needs to be reported as a limitation, not
     absorbed into "the" transfer-gap number — a reviewer at a serious
     venue would reasonably ask exactly this question. Not yet run.

## 5. Key Technical Decisions

**ID / date splitting (in-domain, xbank):** two independent leakage guards,
both already implemented and config-driven (`configs/models/downstream.yaml`'s
`probe:` section):
- *Row → client*: within the train period, train/val is split by unique
  client id (`GroupShuffleSplit`), never by row — a client's several
  monthly rows can't straddle train and val.
- *Time*: test is a held-out **calendar period** (train = all of 2023, test
  = 2024-01/02), not a random row split — the probe is never evaluated on a
  period it could have trained on. The config also warns (not errors) if a
  configured month has zero rows in the target file, since xbank's targets
  are known to skip Nov/Dec 2023 entirely.

**ID / date splitting (transfer, MBD):** client-level folds are already
disjoint by construction (verified above) — inherited as-is rather than
re-split. No separate calendar-cut test set is carved out for MBD, since its
own benchmark protocol is fold-based; this asymmetry with xbank's protocol
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
xbank default.

## 6. Open Items Before Running the Full Comparison

- THP's downstream-probe MLP arm should get feature standardization before
  its first real run through either probe script — its embedding L2 norm
  (~76 median, after an architecture-level fix to an upstream `easy_tpp`
  bug) is still ~10-15x larger than CoLES/Chronos-2's, which could bias
  the MLP arm specifically (LightGBM is scale-invariant, unaffected). Not
  yet added.
- Run the MBD column-slot-assignment ablation described in §4 point 5
  before trusting any MBD transfer number — swap `amount`'s slot
  (`col_11`↔`col_12`) and a couple of the arbitrary categorical mappings,
  check whether downstream probe metrics move meaningfully. Not yet run.
