# Best NER result

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

## 9. Current best: calibrated BIO decoding with model-support filtering

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
