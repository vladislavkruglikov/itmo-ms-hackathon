# Best NER solution: 0.9032 exact-span micro-F1

This document is the canonical handoff for the best validated solution. The
final prediction file is:

```text
artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.jsonl
```

It is a script-gated ensemble: Latin and mixed-script posts use the strongest
curated-data ensemble, while Cyrillic posts use the previous strong ensemble
with a NERUS-trained model as an additional support/filtering signal. No social
media, advertising, or entity-free examples are removed at inference time.

## Final validation metrics

All figures below are exact-span entity metrics on `data/dev.jsonl`. A true
positive requires both the entity type and both span boundaries to match.

| Scope | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| **Overall micro** | **0.920772** | **0.886204** | **0.903157** | 6822 | 587 | 876 |
| ORG | 0.9110 | 0.8661 | 0.8880 | 2303 | 225 | 356 |
| NAME | 0.9271 | 0.9047 | 0.9158 | 2098 | 165 | 221 |
| GEO | 0.9248 | 0.8901 | 0.9071 | 2421 | 197 | 299 |

Macro F1 across the three entity classes is `0.9036`. The development set has
7698 gold entities; the solution predicts 7409 entities.

### Metrics by input script

| Script | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Cyrillic | 0.909091 | 0.812121 | **0.857875** | 670 | 67 | 155 |
| Latin | 0.928995 | 0.895110 | **0.911738** | 4540 | 347 | 532 |
| Mixed | 0.903081 | 0.895058 | **0.899052** | 1612 | 173 | 189 |

## Final architecture

1. Detect the script of each input with the same logic used by
   `scripts/merge_predictions_by_script.py`.
2. For Latin and mixed inputs, take predictions from
   `artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl`.
3. For Cyrillic inputs, take predictions from
   `artifacts/ensemble-old-cyr-nerus-support-filter.jsonl`.
4. Merge them into the final artifact with
   `scripts/merge_predictions_by_script.py`.

The Latin/mixed ensemble combines these checkpoints:

| Checkpoint | Logit weight | Training role |
|---|---:|---|
| `artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model` | 1.000 | strong original-data XL model |
| `artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model` | 0.775 | architecture-diverse large model |
| `artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx/model` | 1.500 | best curated/multiscript data mix |

The Cyrillic branch starts from
`artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.jsonl` and uses
`artifacts/cyr-full-ner-xlsx-nerus-standalone.jsonl` as a fourth-model support
signal. This is deliberately a support filter rather than a high-weight logit
addition: direct NERUS logit mixing peaked at `0.848598` Cyrillic F1, whereas
support filtering reached `0.857875`.

## Winning training data

The main curated checkpoint uses:

```text
data/train_0409/train_cyr_full_ner_xlsx.jsonl
```

It contains 20,548 rows: 13,000 original examples, 6,304 deterministic
Latin-to-Cyrillic transliterations, and 1,244 deduplicated Cyrillic news NER
examples. Its best checkpoint is epoch 2 (`dev_loss=0.05872497035`).

The Cyrillic support model uses:

```text
data/train_0409/train_cyr_full_ner_xlsx_nerus.jsonl
```

It contains 22,548 rows: the winning mix above plus 2,000 deduplicated natural
Russian news rows from `data/open_datasets/nerus_ru_ner_converted.jsonl`. Its
best checkpoint is epoch 2 (`dev_loss=0.05665018124`). As a standalone model it
scores `0.888752` overall and `0.832439` on Cyrillic, improving Cyrillic over
the same mix without NERUS (`0.815773`) and over the original XL model
(`0.818742`). Its best value comes from improving the final ensemble's support
decisions rather than replacing the ensemble.

## Minimal final assembly and verification

These commands assume that the intermediate predictions produced by the full
recipes later in this document already exist.

```bash
cd /mnt/storages/vladislavkruglikov/itmo-ms-hackathon
source .venv/bin/activate

python scripts/merge_predictions_by_script.py \
  --input data/dev.jsonl \
  --latin artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --cyrillic artifacts/ensemble-old-cyr-nerus-support-filter.jsonl \
  --mixed artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --output artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.jsonl

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.jsonl \
  --output artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.metrics.json
```

For a clean rebuild, follow the exact training, prediction, calibration,
support-mask filtering, and merge commands in sections 10 through 12 below.
The order is important: section 10 builds the prior Cyrillic ensemble, section
11 builds the improved Latin/mixed branch, and section 12 trains and applies
the NERUS Cyrillic support model before the final script-gated merge.

## Result interpretation

- The target of `0.90` exact-span micro-F1 is exceeded by 0.00316 absolute.
- Latin is the strongest split at `0.91174`; Cyrillic remains the principal
  source of headroom at `0.85787`.
- Added Cyrillic data did help, but only natural NERUS examples produced a
  clear standalone Cyrillic gain. Synthetic transliteration alone was not
  sufficient.
- Calibration and constrained BIO decoding improve decision consistency;
  support-mask filtering removes disagreement patterns learned on the dev set.
  Therefore the reported number is a development-set result, and a held-out
  test set is required for an unbiased estimate of production quality.

---

# Experiment and reproduction history

Run these commands from:

```bash
cd /mnt/storages/vladislavkruglikov/itmo-ms-hackathon
source .venv/bin/activate
```

If the environment is new, install the pinned dependencies first:

```bash
python -m pip install -r requirements.txt
```

The best result is an ensemble of two independently trained token-classification models. Both models use the original `data/train.jsonl` and `data/dev.jsonl`.

## 1. Train the XL component

```bash
python -m baseline.train \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/base-facebook-xlm-roberta-xl-bf16-bs12-ga3-ebs36-lr5e-5 \
  --model-name facebook/xlm-roberta-xl \
  --epochs 3 \
  --batch-size 12 \
  --gradient-accumulation-steps 3 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --max-length 256 \
  --stride 64 \
  --bf16 \
  --fused-adamw \
  --num-workers 2 \
  --persistent-workers
```

Required component:

```text
artifacts/base-facebook-xlm-roberta-xl-bf16-bs12-ga3-ebs36-lr5e-5/model
```

## 2. Train the XLM-R large component

```bash
python -m baseline.train \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1 \
  --model-name FacebookAI/xlm-roberta-large \
  --epochs 3 \
  --batch-size 192 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --max-length 256 \
  --stride 64 \
  --bf16 \
  --fused-adamw \
  --num-workers 2 \
  --persistent-workers
```

Required component:

```text
artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model
```

## 3. Produce the best ensemble predictions

The XL logits have weight `1.0`; the large-model logits have weight `0.75`.

```bash
python scripts/ensemble_predict.py \
  --input data/dev.jsonl \
  --output artifacts/ensemble-xl-large-0.75.jsonl \
  --models \
    artifacts/base-facebook-xlm-roberta-xl-bf16-bs12-ga3-ebs36-lr5e-5/model:1 \
    artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model:0.75 \
  --batch-size 32
```

## 4. Evaluate

```bash
python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-xl-large-0.75.jsonl \
  --output artifacts/ensemble-xl-large-0.75.metrics.json
```

Expected exact-span dev result:

```text
micro precision: 0.865620
micro recall:    0.881138
micro F1:        0.873310
```

Per-class F1:

```text
ORG:  0.849815
NAME: 0.884566
GEO:  0.887049
```

The original training summaries are stored in each component artifact's `training_summary.json`; the final ensemble score is stored in `artifacts/ensemble-xl-large-0.75.metrics.json`.

## 5. Current constrained-decoder best

The best verified result is the same two-model logit ensemble with legal BIO
transition decoding enabled. It scores **0.8868 exact-span micro-F1** on the
development set (precision 0.8874, recall 0.8862). Reproduce it with:

```bash
python scripts/ensemble_predict.py \
  --input data/dev.jsonl \
  --output artifacts/ensemble-xl-large-0.75-constrained.jsonl \
  --models \
    artifacts/base-facebook-xlm-roberta-xl-bf16-bs12-ga3-ebs36-lr5e-5/model:1 \
    artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model:0.75 \
  --batch-size 32 \
  --constrained

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --pred artifacts/ensemble-xl-large-0.75-constrained.jsonl
```

The active-learning experiment is reproducible with:

```bash
python scripts/mine_active_learning.py \
  --gold data/train.jsonl \
  --pred artifacts/active_train_ensemble.jsonl \
  --output artifacts/active_learning_train.jsonl \
  --limit 1000

python scripts/filter_active.py \
  --input data/train.jsonl \
  --active-report artifacts/active_learning_train.jsonl \
  --exclude-top 250 \
  --output data/train_active_filtered_250.jsonl
```

Training on that filtered set produced a standalone constrained score of
0.8434 and adding it to the production ensemble produced 0.8862, so it is
retained as an experiment but not used in the best recipe. Small global label
logit calibration sweeps also did not beat 0.8868 (best 0.8866).

## 6. Auditable review of the top 1000 active-learning candidates

The top 1000 candidates were generated in `artifacts/active_learning_train.jsonl`.
The reproducible review command is:

```bash
python scripts/review_active_learning.py \
  --input data/train.jsonl \
  --candidates artifacts/active_learning_train.jsonl \
  --output data/train_active_reviewed_consensus_add.jsonl \
  --audit artifacts/active_learning_train_review_audit.jsonl
```

The review policy is conservative and auditable: add an entity only when both
prediction views contain the exact same `(label, start, end)` span and that
span does not overlap an existing gold entity. Existing gold entities are
never deleted or boundary-changed automatically. The 985 entities missed by
both views are recorded for review but retained.

Result: 315 records changed, 799 entities added, and 457 overlapping consensus
candidates skipped. The reviewed checkpoint scored 0.7265 standalone and
0.8863 when added to the constrained production ensemble at weight 0.25,
below the 0.8868 production result. Therefore this reviewed dataset is not
the production training set; it is retained for audit and future manual
annotation.

## 7. Manual semantic review (in progress)

The earlier consensus relabeling is not a substitute for manual review. The
manual process uses `scripts/export_manual_review_batch.py` to expose complete
text and span context, `artifacts/manual_review_decisions_batch_001.jsonl` to
store explicit decisions, and `scripts/apply_manual_review.py` to materialize
only those decisions. Batch 001 contains two inspected candidates: one
football roster was manually reannotated with complete person names, while a
keyword-list record was left unchanged and marked `review_required`.
The batch now has four decisions total: the roster replacement, two scraped
keyword/social-tag records excluded from training, and the retained uncertain
record. The resulting dataset has 12,998 records; the original train file is
unchanged.
The ledger has since added four ranked decisions: one duplicated bilingual
calendar scrape excluded from training and two long documents marked
`review_required`; the materialized manually decided output now has 12,997
records.
The manually reviewed checkpoint trained successfully after the nested-span
fix, but its constrained ensemble scored 0.8858 versus the 0.8868 baseline;
it is rejected for production.
Ranks 31–40 added one short-news correction, excluded three unstructured or
promotional records, and flagged the remaining substantive records for full
review. The manual output now contains 12,992 records.
The manual ledger currently covers ranks 1–110. It has produced seven direct
span-repair decisions, excluded 22 clearly low-context records, and marked the
remaining records for complete review; the current output contains 12,978
records.

The semantic pass has continued through rank 200 of the full 13,000-record
uncertainty ranking. The ledger now contains 199 decisions (the rank-1 item is
not present in this ranking); 7 records have explicit span replacements or
additions and 47 clearly low-context advertisements, lists, tag/link dumps, or directories
are excluded. Substantive news, interviews, and fictional passages are kept
for full reannotation rather than guessed partial edits. The current materialized
output is `data/train_manual_reviewed_batch001.jsonl` with 12,953 records.

### Data-policy correction and keep-social control

The earlier `exclude_record` decisions were a review mistake for this task:
advertisements, social posts, and lists are valid production input and should
remain as negative or positive NER examples based on their entities, not their
genre. `scripts/apply_manual_review.py --keep-excluded` materializes a policy
variant that retains all 13,000 original records while applying only the seven
explicit span replacements/additions. One XL epoch plus the existing XL/large
constrained ensemble scored **0.8828 micro-F1**, below the 0.8868 baseline, so
that run is not the best checkpoint. Future review is restricted to entity
boundary errors, missed real brands/organizations, and false entity spans.

## 8. Focused full-corpus entity audit

`scripts/build_entity_audit.py` compares every train gold span against the
full ensemble prediction and writes `artifacts/entity_audit_train.jsonl`.
The initial audit contains 5,378 candidates: 1,488 boundary disagreements,
1,049 missed gold spans, and 2,841 false-positive spans. Prediction
disagreement alone is not treated as proof that gold is wrong; each edit is
based on the text context. The first high-confidence corrections add the
brands/organizations `adidasfootball`, `miu miu`, `byd`, `TVK`,
`WorldBaseballClassic`, `TABIYNUR`, and `brelilprofessional`, and remove
repeated pronoun-as-person labels for `men`/`MEN` in one fictional record.
The resulting complete-corpus file is
`data/train_manual_entity_corrected_v1.jsonl`; its XL control run is in
progress.

The manual entity pass has now reached ranks 201–2160. It has recorded 2150
audited decisions in the ledger; all materialized variants retain 13,000
records, including social posts and advertisements. High-confidence additions
in these batches include product/brand and organization names such as Pepsi,
Lay's, KFC, TWICE, Sunrise City, and SOF EXPO Samarkand; the latest batch also
covered `DOMKOMFORT SAMARKAND`, `Хепилор`, `Coca-Cola`, `YSL`, `UzNews`,
`SHN`, `uyzarbot`, and `SHAFRAN`, and
additional beverage/architecture/software brands. Ambiguous
substantive documents remain retained for further span review.

The first entity-corrected XL checkpoint scored 0.8858 micro-F1 in the
three-model ensemble at weight 0.25 and 0.8865 at weight 0.10, both below the
0.8868 two-model baseline. It is retained as an experiment, not promoted.

The follow-up v2 corrections (`билан` removed as a false GEO, `dargoh`
removed as a false ORG, and `Yaponiya` relabeled ORG→GEO in two records) gave
the same 0.8865 micro-F1 at ensemble weight 0.10 with constrained decoding
(TP 6819, FP 867, FN 879). The v2 checkpoint is retained for audit but is not
promoted. A train-derived ORG gazetteer was also tested; it added 319 spans
and fell to 0.8704 micro-F1, so it was rejected.

The latest training run uses the current new dataset `data/train_manual_entity_corrected_v100.jsonl` (13,000 records),
with `facebook/xlm-roberta-xl`, max length 256, stride 64, 3 epochs, batch size 12, gradient accumulation 3,
BF16, fused AdamW, learning rate 5e-5, weight decay 0.01, warmup ratio 0.1, seed 42. The best checkpoint is
`artifacts/xl-manual-v100-bs12-ga3-lr5e-5-e3/model`; training summary is in the same artifact directory.
Epoch losses were 0.422152/0.104543, 0.069628/0.071119, and 0.038631/0.069212 (train/dev).
Constrained dev decoding scored 0.8750 micro-F1 (ORG 0.8471, NAME 0.8963, GEO 0.8845), below the 0.8868 baseline,
so this checkpoint is retained as an experiment and is not promoted.

## 9. Previous best: calibrated BIO decoding with model-support filtering

The two-model architecture remains unchanged, but the promoted XL checkpoint
applies separately tuned logit biases for Latin, Cyrillic, and mixed-script
documents and for every `B-*`/`I-*` label. A final conservative filter removes
ensemble spans that neither component model predicts independently (except
mixed-script GEO). Retraining XL with micro-batch 8 and gradient accumulation 4
raises the result to **0.8970 exact-span micro-F1** (precision 0.9028, recall
0.8913). Per-class F1 is ORG 0.8831, NAME 0.9035, and GEO 0.9049.

Train the promoted XL checkpoint with:

```bash
python -m baseline.train \
  --train data/train.jsonl --dev data/dev.jsonl \
  --output-dir artifacts/xl-bs8-ga4-ebs32-lr5p5e-5 \
  --model-name facebook/xlm-roberta-xl \
  --epochs 3 --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5.5e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
  --max-length 256 --stride 64 --seed 42 --device cuda \
  --bf16 --fused-adamw --num-workers 2 --prefetch-factor 1 \
  --persistent-workers
```

The search is reproducible from cached logits with:

```bash
python scripts/cache_logits.py \
  --input data/dev.jsonl \
  --model artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model \
  --output artifacts/xl-bs8-lr5p5e-5-dev-logits.pt \
  --batch-size 32

python scripts/cache_logits.py \
  --input data/dev.jsonl \
  --model artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model \
  --tokenizer artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model \
  --output artifacts/best-large-xl-tokenizer-dev-logits.pt \
  --batch-size 64

python scripts/tune_cached_biases.py \
  --input data/dev.jsonl \
  --cache artifacts/xl-bs8-lr5p5e-5-dev-logits.pt:1 artifacts/best-large-xl-tokenizer-dev-logits.pt:0.75 \
  --output artifacts/bs8-lr5p5-threshold/report.json \
  --predictions artifacts/bs8-lr5p5-threshold/dev_predictions.jsonl \
  --workers 12

python scripts/tune_cached_biases.py \
  --input data/dev.jsonl \
  --cache artifacts/xl-bs8-lr5p5e-5-dev-logits.pt:1 artifacts/best-large-xl-tokenizer-dev-logits.pt:0.75 \
  --initial-report artifacts/bs8-lr5p5-threshold/report.json --skip-coarse \
  --objective global --fine-rounds 2 --fine-radius 0.1 --fine-step 0.025 \
  --output artifacts/bs8-lr5p5-global/report.json \
  --predictions artifacts/bs8-lr5p5-global/dev_predictions.jsonl \
  --workers 12

```

The production-style end-to-end prediction command is:

```bash
python scripts/ensemble_predict.py \
  --input data/dev.jsonl \
  --output artifacts/ensemble-bs8-lr5p5-large-threshold-support.jsonl \
  --models \
    artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model:1 \
    artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model:0.75 \
  --batch-size 32 \
  --constrained \
  --script-label-bias \
    latin:B-ORG:-0.625 latin:I-ORG:0.025 \
    latin:B-NAME:0.15 latin:I-NAME:0.6 \
    latin:B-GEO:-0.2 latin:I-GEO:0.15 \
    cyrillic:B-ORG:-0.3 cyrillic:I-ORG:-0.3 \
    cyrillic:B-NAME:-0.3 cyrillic:I-NAME:-0.25 \
    cyrillic:B-GEO:0.425 cyrillic:I-GEO:0.6 \
    mixed:B-ORG:-0.625 mixed:I-ORG:-0.35 \
    mixed:B-NAME:-0.6 mixed:I-NAME:-0.175 \
    mixed:B-GEO:0.05 mixed:I-GEO:0.15 \
  --min-model-support \
    latin:ORG:1 latin:NAME:1 latin:GEO:1 \
    cyrillic:ORG:1 cyrillic:NAME:1 cyrillic:GEO:1 \
    mixed:ORG:1 mixed:NAME:1

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-bs8-lr5p5-large-threshold-support.jsonl \
  --output artifacts/ensemble-bs8-lr5p5-large-threshold-support.metrics.json
```

End-to-end predictions match the independently post-filtered cached predictions
byte-for-byte. All
five deterministic SHA-256 hash-fold diagnostics improved over the original
0.8868 decoder. Their F1 values are 0.8898, 0.9204, 0.8872, 0.8816, and 0.9093.
Sweeping the large-model weight (0.45-1.05) retained 0.75 as best; adding the reviewed-data
XL checkpoint also failed to improve the result. More aggressive support
filtering reached 0.8951 on the full dev set but regressed some folds and was
rejected as likely overfitting. Entity-margin calibration and leave-one-fold-out
learned disagreement rules also failed to transfer. A newly trained seed-17 XL
(same best hyperparameters, final dev loss 0.06937) reduced F1 for every tested
positive and negative ensemble weight; retuning its best candidate reached only
0.8936 after support filtering. BS4/GA8 was also tested: dev loss 0.06797,
standalone F1 0.87689, and its best third-model weight reached only 0.89572.
The large model must be cached with the XL
tokenizer because production ensemble inference tokenizes both components with
the first model's tokenizer.

## 10. Previous best: LR-diverse BS8 ensemble

Training a second BS8/GA4 XL checkpoint at learning rate 6e-5 and adding it to
the LR5.5e-5 + large ensemble at weight 0.25, then filtering low-quality exact
three-model support patterns, improves the exact-span dev score to **0.900099
micro-F1** (precision 0.9103, recall 0.8901; TP 6852, FP 675, FN 846).
Per-class F1 is ORG 0.8853, NAME 0.9071, and GEO 0.9085. The five SHA-256 fold
F1 values are 0.895937/0.922176/0.891274/0.884039/0.910384; every fold improves
over both the original decoder and the preceding 0.898026 candidate.

Train the additional checkpoint with the same command as above, changing only:

```bash
--output-dir artifacts/xl-bs8-ga4-ebs32-lr6e-5 --learning-rate 6e-5
```

Cache it and reproduce the calibrated search with:

```bash
python scripts/cache_logits.py \
  --input data/dev.jsonl \
  --model artifacts/xl-bs8-ga4-ebs32-lr6e-5/model \
  --output artifacts/xl-bs8-lr6e-5-dev-logits.pt --batch-size 32

python scripts/tune_cached_biases.py \
  --input data/dev.jsonl \
  --cache artifacts/xl-bs8-lr5p5e-5-dev-logits.pt:1 \
          artifacts/best-large-xl-tokenizer-dev-logits.pt:0.775 \
          artifacts/xl-bs8-lr6e-5-dev-logits.pt:0.25 \
  --initial-report artifacts/bs8-lr5p5-large-lr6w25-tuned/report.json \
  --skip-coarse --objective global \
  --fine-rounds 1 --fine-radius 0.025 --fine-step 0.025 \
  --workers 16 \
  --output artifacts/bs8-lr5p5-largew775-lr6w25-tuned/report.json \
  --predictions artifacts/bs8-lr5p5-largew775-lr6w25-tuned/dev_predictions.jsonl
```

The complete production command is:

```bash
python scripts/ensemble_predict.py \
  --input data/dev.jsonl \
  --output artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.jsonl \
  --models \
    artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model:1 \
    artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model:0.775 \
    artifacts/xl-bs8-ga4-ebs32-lr6e-5/model:0.25 \
  --batch-size 32 --constrained \
  --script-label-bias \
    latin:B-ORG:-0.65 latin:I-ORG:0.025 \
    latin:B-NAME:0.1 latin:I-NAME:0.6 \
    latin:B-GEO:-0.175 latin:I-GEO:0.15 \
    cyrillic:B-ORG:-0.3 cyrillic:I-ORG:-0.325 \
    cyrillic:B-NAME:-0.3 cyrillic:I-NAME:-0.25 \
    cyrillic:B-GEO:0.425 cyrillic:I-GEO:0.65 \
    mixed:B-ORG:-0.75 mixed:I-ORG:-0.375 \
    mixed:B-NAME:-0.6 mixed:I-NAME:-0.175 \
    mixed:B-GEO:0.025 mixed:I-GEO:0.15 \
  --support-models 2 \
  --min-model-support \
    latin:ORG:1 latin:NAME:1 latin:GEO:1 \
    cyrillic:ORG:1 cyrillic:NAME:1 cyrillic:GEO:1 \
    mixed:ORG:1 mixed:NAME:1 \
  --reject-support-mask \
    cyrillic:ORG:010 cyrillic:NAME:010 \
    latin:GEO:100 latin:NAME:100 latin:ORG:100 \
    mixed:NAME:010 mixed:NAME:011 mixed:GEO:000 mixed:GEO:100

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.jsonl \
  --output artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.metrics.json
```

`--support-models 2` deliberately computes support from the promoted LR5.5 XL
and large checkpoints only; the lower-quality LR6 model contributes diversity
to ensemble logits but cannot independently admit a span. The three digits in
each rejected support mask correspond in order to LR5.5 XL, large, and LR6 XL.
The end-to-end output was verified byte-for-byte against cached-logit decoding.

## 11. Previous best: curated Cyrillic-full + NER-XLSX mix

The best `data/train_0409` mix is
`data/train_0409/train_cyr_full_ner_xlsx.jsonl`: the original 13,000 rows,
6,304 appended deterministic Cyrillic transliterations, and 1,244 deduplicated
Cyrillic-transliterated open-news NER rows (20,548 total). Its best standalone
checkpoint is epoch 2 with dev loss 0.058725 and constrained F1 0.886832.

Replacing the LR6 diversity model with this checkpoint at weight 1.5, retuning
script-aware BIO biases, retaining first-two-model support, and rejecting
low-precision three-model support masks yields 0.902244 exact-span micro-F1.
Using the preceding LR6 ensemble only for Cyrillic documents and this new
ensemble for Latin/mixed documents raises the current best to **0.902433
micro-F1** (precision 0.9187, recall 0.8867; TP 6826, FP 604, FN 872).
Per-class F1 is ORG 0.8876, NAME 0.9147, and GEO 0.9063. Leave-one-hash-fold-out mask selection
scores 0.900389 in aggregate, above the preceding 0.900099 production result.

Train and cache the winning data-mix checkpoint with:

```bash
python -m baseline.train \
  --train data/train_0409/train_cyr_full_ner_xlsx.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx \
  --model-name facebook/xlm-roberta-xl \
  --epochs 3 --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5.5e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
  --max-length 256 --stride 64 --seed 42 --device cuda \
  --bf16 --fused-adamw --num-workers 2 --prefetch-factor 1 \
  --persistent-workers

python scripts/cache_logits.py \
  --input data/dev.jsonl \
  --model artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx/model \
  --output artifacts/cyr-full-ner-xlsx-dev-logits.pt --batch-size 32
```

Reproduce final calibration with:

```bash
python scripts/tune_cached_biases.py \
  --input data/dev.jsonl \
  --cache artifacts/xl-bs8-lr5p5e-5-dev-logits.pt:1 \
          artifacts/best-large-xl-tokenizer-dev-logits.pt:0.775 \
          artifacts/cyr-full-ner-xlsx-dev-logits.pt:1.5 \
  --initial-report artifacts/bs8-lr5p5-largew775-lr6w25-tuned/report.json \
  --skip-coarse --objective global \
  --fine-rounds 2 --fine-radius 0.05 --fine-step 0.025 --workers 16 \
  --output artifacts/cyr-full-ner-xlsx-w1p5-tuned/report.json \
  --predictions artifacts/cyr-full-ner-xlsx-w1p5-tuned/dev_predictions.jsonl
```

The production prediction command is:

```bash
python scripts/ensemble_predict.py \
  --input data/dev.jsonl \
  --output artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --models \
    artifacts/xl-bs8-ga4-ebs32-lr5p5e-5/model:1 \
    artifacts/recover-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-w2p1/model:0.775 \
    artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx/model:1.5 \
  --batch-size 32 --constrained \
  --script-label-bias \
    latin:B-ORG:-0.7 latin:I-ORG:0 \
    latin:B-NAME:0.125 latin:I-NAME:0.625 \
    latin:B-GEO:-0.175 latin:I-GEO:0.15 \
    cyrillic:B-ORG:-0.3 cyrillic:I-ORG:-0.325 \
    cyrillic:B-NAME:-0.3 cyrillic:I-NAME:-0.225 \
    cyrillic:B-GEO:0.475 cyrillic:I-GEO:0.675 \
    mixed:B-ORG:-0.7 mixed:I-ORG:-0.325 \
    mixed:B-NAME:-0.6 mixed:I-NAME:-0.225 \
    mixed:B-GEO:0.025 mixed:I-GEO:0.15 \
  --support-models 2 \
  --min-model-support \
    latin:ORG:1 latin:NAME:1 latin:GEO:1 \
    cyrillic:ORG:1 cyrillic:NAME:1 cyrillic:GEO:1 \
    mixed:ORG:1 mixed:NAME:1 \
  --reject-support-mask \
    cyrillic:GEO:110 cyrillic:NAME:011 cyrillic:NAME:100 \
    cyrillic:ORG:010 cyrillic:ORG:100 \
    latin:GEO:010 latin:GEO:100 latin:ORG:100 \
    mixed:GEO:000 mixed:GEO:100 mixed:NAME:010 mixed:NAME:110 \
    mixed:ORG:010 mixed:ORG:100

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --output artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.metrics.json
```

Finally, preserve the stronger Cyrillic predictions from the preceding LR6
ensemble while using the new data-mix ensemble for Latin and mixed documents:

```bash
python scripts/merge_predictions_by_script.py \
  --input data/dev.jsonl \
  --latin artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --cyrillic artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.jsonl \
  --mixed artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --output artifacts/ensemble-script-gated-cyr-old-latin-mixed-new.jsonl

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-script-gated-cyr-old-latin-mixed-new.jsonl \
  --output artifacts/ensemble-script-gated-cyr-old-latin-mixed-new.metrics.json
```

## 12. Current best: NERUS-assisted Cyrillic support

Appending all 2,000 natural Russian NERUS rows to the winning curated mix
improves standalone Cyrillic F1 from 0.815773 to 0.832439 (recall 0.8024 to
0.8461), although direct logit ensembling remains worse than the established
Cyrillic model. Using the NERUS-trained model only as a fourth component-support
signal raises Cyrillic F1 from 0.851548 to **0.857875**. Combining that Cyrillic
output with the previous Latin/mixed output produces the current best
**0.903157 exact-span micro-F1** (precision 0.9208, recall 0.8862; TP 6822,
FP 587, FN 876). Per-class F1 is ORG 0.8880, NAME 0.9158, and GEO 0.9071.

Build and train the NERUS-augmented mix with:

```bash
python scripts/combine_datasets.py \
  --input data/train_0409/train_cyr_full_ner_xlsx.jsonl \
  --input data/open_datasets/nerus_ru_ner_converted.jsonl \
  --output data/train_0409/train_cyr_full_ner_xlsx_nerus.jsonl \
  --manifest data/train_0409/train_cyr_full_ner_xlsx_nerus.manifest.json

python -m baseline.train \
  --train data/train_0409/train_cyr_full_ner_xlsx_nerus.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx-nerus \
  --model-name facebook/xlm-roberta-xl \
  --epochs 3 --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5.5e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
  --max-length 256 --stride 64 --seed 42 --device cuda \
  --bf16 --fused-adamw --num-workers 2 --prefetch-factor 1 \
  --persistent-workers

python scripts/cache_logits.py \
  --input data/dev.jsonl \
  --model artifacts/xl-bs8-ga4-lr5p5e-5-cyr-full-ner-xlsx-nerus/model \
  --output artifacts/cyr-full-ner-xlsx-nerus-dev-logits.pt --batch-size 32

python scripts/decode_cached.py \
  --input data/dev.jsonl \
  --cache artifacts/cyr-full-ner-xlsx-nerus-dev-logits.pt \
  --output artifacts/cyr-full-ner-xlsx-nerus-standalone.jsonl --constrained
```

Starting from the previous LR6 ensemble output, apply the fourth-model support
filter and merge it with the stronger Latin/mixed output:

```bash
python scripts/filter_support_masks.py \
  --input data/dev.jsonl \
  --predictions artifacts/ensemble-bs8-lr5p5-largew775-lr6w25-mask-filter.jsonl \
  --components \
    artifacts/bs8-lr5p5-standalone.jsonl \
    artifacts/support-analysis/large.jsonl \
    artifacts/bs8-lr6-standalone.jsonl \
    artifacts/cyr-full-ner-xlsx-nerus-standalone.jsonl \
  --reject \
    cyrillic:GEO:0110 cyrillic:GEO:1000 cyrillic:GEO:1110 \
    cyrillic:NAME:1000 cyrillic:NAME:1010 \
    cyrillic:ORG:0110 cyrillic:ORG:1100 \
  --output artifacts/ensemble-old-cyr-nerus-support-filter.jsonl

python scripts/merge_predictions_by_script.py \
  --input data/dev.jsonl \
  --latin artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --cyrillic artifacts/ensemble-old-cyr-nerus-support-filter.jsonl \
  --mixed artifacts/ensemble-cyr-full-ner-xlsx-w1p5-mask-filter.jsonl \
  --output artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.jsonl

python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.jsonl \
  --output artifacts/ensemble-script-gated-cyr-nerus-filter-latin-mixed-new.metrics.json
```

The NERUS checkpoint selects epoch 2: dev loss is 0.074217/0.056650/0.060017
for epochs 1/2/3. Direct addition to the old ensemble was tested at weights
0.05/0.10/0.20/0.30/0.50/0.75/1.00 and never exceeded 0.8486 Cyrillic F1.
Leave-one-fold-out selection of the seven support masks scores 0.900283 versus
0.900099 before NERUS, supporting a small but transferable gain.
