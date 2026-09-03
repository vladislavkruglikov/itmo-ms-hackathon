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
