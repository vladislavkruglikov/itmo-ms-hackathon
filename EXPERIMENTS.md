# Эксперименты

Этот файл — реестр экспериментов обучения NER-моделей. Основная метрика качества —
`exact-span micro-F1` на `data/dev.jsonl`; сущность засчитывается только при полном
совпадении `hash`, класса и символьных границ.

## Общие условия

- GPU: NVIDIA H100 80 GB HBM3, один ускоритель.
- Обучающая выборка по умолчанию: `train` (13 000 документов); для варианта `train_v2` это явно отмечено в таблице. Число окон зависит от tokenizer модели.
- Валидационная выборка: 1 500 документов; число окон зависит от tokenizer модели.
- `max_length=256`, `stride=64`, `seed=42`.
- PyTorch 2.6.0+cu124, Transformers 5.14.1; FlashAttention 2.7.4.post1 там,
  где явно указан FA2.
- `weight_decay=0.01`, `warmup_ratio=0.1`, линейный decay learning rate,
  `max_grad_norm=1.0`.
- Если не указано иное: 3 эпохи, BF16 autocast, gradient accumulation 1,
  `persistent_workers=true` при `num_workers > 0`.
- Время включает train/dev-loss и сохранение лучшей модели. Compute time исключает
  валидацию и сохранение модели.
- `—` означает, что значение не было рассчитано или соответствующий артефакт был
  удалён. Значения не восстанавливаются приблизительно.

## Полные эксперименты с сохранёнными результатами

| Run | Dataset version | Model | BS | LR | Epochs | Attention / weights | AdamW | Batching | Workers / prefetch | Best dev loss | Micro-F1 | Macro-F1 | Latin micro-F1 | Latin macro-F1 | Cyrillic micro-F1 | Cyrillic macro-F1 | Train time, s | Compute, s | HFU |
|---|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xl-bs12-ga3-ebs36-lr5e-5` | `train` |  `facebook/xlm-roberta-xl` | 12 | 5e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA3 (effective BS36) | 2 / 1 | 0.06758 | **0.86020** | **0.86100** | **0.87428** | **0.87539** | **0.81696** | **0.81661** | 917.21 | 830.65 | 18.54% |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-8-e1` | `train_v2` | `facebook/xlm-roberta-xl` | 12 | 1e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA3 (effective BS36); initialized from best XL checkpoint | 2 / 1 | 0.06753 | **0.85977** | **0.86055** | — | — | — | — | 800.11 | 771.32 | 18.87% |
| `xl-best-to-v3-bs12-ga3-ebs36-lr1e-8-e1` | `train_v3` | `facebook/xlm-roberta-xl` | 12 | 1e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA3 (effective BS36); initialized from best XL checkpoint | 2 / 1 | 0.06755 | 0.85959 | 0.86038 | — | — | — | — | 805.67 | 777.00 | 18.73% |
| `xl-bs16-ga2-ebs32-lr5e-5` | `train` |  `facebook/xlm-roberta-xl` | 16 | 5e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA2 (effective BS32) | 2 / 1 | 0.06750 | 0.85780 | 0.85880 | — | — | — | — | 885.52 | 802.49 | 19.19% |
| `xl-bs28-lr5e-5` | `train` |  `facebook/xlm-roberta-xl` | 28 | 5e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | **0.06507** | **0.85609** | **0.85708** | — | — | — | — | 836.65 | 750.88 | 20.51% |
| `large-best-to-v2-bs160-ga2-ebs320-lr3e-8-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 160 | 3e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA2 (effective BS320); initialized from best large checkpoint | 2 / 1 | 0.06897 | **0.85536** | **0.85603** | — | — | — | — | 102.20 | 97.49 | 24.01% |
| `large-best-to-v2-bs256-lr3e-8-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 256 | 3e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06878 | **0.85525** | **0.85593** | — | — | — | — | **97.45** | **92.60** | **25.28%** |
| `large-best-to-v2-bs192-lr3e-8-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 192 | 3e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06838 | **0.85500** | **0.85566** | — | — | — | — | 101.57 | 96.79 | 24.18% |
| `large-best-to-v2-bs192-lr3e-8-e2` | `train_v2` | `FacebookAI/xlm-roberta-large` | 192 | 3e-8 | 2 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06692 | 0.85495 | 0.85551 | — | — | — | — | 202.95 | 193.62 | 24.18% |
| `xl-bs16-ga2-ebs32-lr7e-5` | `train` |  `facebook/xlm-roberta-xl` | 16 | 7e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA2 (effective BS32) | 2 / 1 | **0.06496** | 0.85480 | 0.85520 | — | — | — | — | 868.24 | 802.34 | 19.19% |
| `large-bs192-lr2e-4-w2p1` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.07018 | **0.85450** | **0.85530** | — | — | — | — | 117.06 | 104.13 | 23.78% |
| `large-best-to-v2-bs128-lr3e-8-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 128 | 3e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06770 | 0.85421 | 0.85484 | — | — | — | — | 104.47 | 99.69 | 23.48% |
| `large-best-to-v2-bs192-lr5e-8-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 192 | 5e-8 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06738 | 0.85415 | 0.85473 | — | — | — | — | 101.46 | 96.69 | 24.20% |
| `xl-bs16-ga2-ebs32-lr4e-5` | `train` |  `facebook/xlm-roberta-xl` | 16 | 4e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA2 (effective BS32) | 2 / 1 | 0.06954 | 0.85200 | 0.85280 | — | — | — | — | 891.71 | 804.09 | 19.15% |
| `large-best-to-v2-bs192-lr3e-8-e3` | `train_v2` | `FacebookAI/xlm-roberta-large` | 192 | 3e-8 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06593 | 0.85105 | 0.85151 | — | — | — | — | 304.11 | 290.06 | 24.21% |
| `large-best-to-v2-bs192-lr1e-7-e1` | `train_v2` |  `FacebookAI/xlm-roberta-large` | 192 | 1e-7 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.06568 | 0.85047 | 0.85097 | — | — | — | — | 101.67 | 96.85 | 24.17% |
| `large-bs192-lr2e-4-sortish20` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | sortish pool ×20 | 2 / 1 | **0.06840** | 0.84848 | 0.84875 | — | — | — | — | **70.26** | **57.22** | **43.27%** |
| `large-bs192-lr1.5e-4-fa-varlen` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 1.5e-4 | 3 | FA2 varlen, direct BF16 weights | fused | dynamic padding | 2 / 1 | **0.06342** | 0.84719 | 0.84873 | — | — | — | — | 102.44 | 95.19 | 26.01% |
| `xl-bs12-ga4-ebs48-lr5e-5` | `train` |  `facebook/xlm-roberta-xl` | 12 | 5e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA4 (effective BS48) | 2 / 1 | 0.07619 | 0.84670 | 0.84750 | — | — | — | — | 906.91 | 815.06 | 18.89% |
| `large-bs192-lr2e-4-fa-varlen-unfused` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | FA2 varlen, FP32 weights + BF16 AMP | unfused | dynamic padding | 2 / 1 | 0.07142 | 0.84463 | 0.84553 | — | — | — | — | 113.47 | 99.69 | 24.84% |
| `large-bs192-lr2e-4-fa-varlen` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | FA2 varlen, direct BF16 weights | fused | dynamic padding | 2 / 1 | 0.06447 | 0.84041 | 0.84135 | — | — | — | — | 102.13 | 94.67 | 26.15% |
| `large-bs192-lr2e-4-sortish5` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | sortish pool ×5 | 2 / 1 | 0.07158 | 0.83975 | 0.84057 | — | — | — | — | 77.67 | 64.39 | 38.45% |
| `large-best-to-v2-bs192-lr1.5e-4-e1` | `train_v2` | `FacebookAI/xlm-roberta-large` | 192 | 1.5e-4 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.07496 | 0.83238 | 0.83295 | — | — | — | — | 101.93 | 96.78 | 24.18% |
| `xl-bs28-lr2e-5` | `train` |  `facebook/xlm-roberta-xl` | 28 | 2e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.08878 | 0.82219 | 0.82351 | — | — | — | — | 825.53 | 746.55 | 20.63% |
| `large-best-to-v2-bs192-lr1e-6-e1` | `train_v2` |  `FacebookAI/xlm-roberta-large` | 192 | 1e-6 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from best large checkpoint | 2 / 1 | 0.07325 | 0.81325 | 0.81248 | — | — | — | — | 101.26 | 96.70 | 24.20% |
| `large-best-to-train-v2-bs192-lr2e-5` | `train_v2` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding; initialized from `large-bs192-lr2e-4-w2p1` | 2 / 1 | 0.07887 | 0.79518 | 0.79702 | — | — | — | — | 296.44 | 289.44 | 24.26% |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-6-e1` | `train_v2` | `facebook/xlm-roberta-xl` | 12 | 1e-6 | 1 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding, GA3 (effective BS36); initialized from best XL checkpoint | 2 / 1 | 0.08248 | 0.73147 | 0.73425 | — | — | — | — | 803.00 | 774.14 | 18.80% |
| `tahrirchi-bs512-lr2.5e-4` | `train` |  `tahrirchi/tahrirchi-bert-base` | 512 | 2.5e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.13498 | **0.71972** | **0.72240** | — | — | — | — | 40.18 | 36.36 | 13.54% |
| `tahrirchi-bs512-lr2e-4` | `train` |  `tahrirchi/tahrirchi-bert-base` | 512 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | **0.13074** | 0.71730 | 0.71969 | — | — | — | — | 40.56 | 36.58 | 13.46% |
| `tahrirchi-bs512-lr3e-4` | `train` |  `tahrirchi/tahrirchi-bert-base` | 512 | 3e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.13310 | 0.71421 | 0.71656 | — | — | — | — | 40.39 | 36.44 | 13.51% |
| `tahrirchi-bs512-lr2.5e-4-e5` | `train` |  `tahrirchi/tahrirchi-bert-base` | 512 | 2.5e-4 | 5 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.13297 | 0.70064 | 0.70366 | — | — | — | — | 64.75 | 60.44 | 13.58% |
| `m-distilbert-bs1024-lr2e-4` | `train` |  `distilbert/distilbert-base-multilingual-cased` | 1024 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | **0.11970** | **0.68568** | **0.68919** | — | — | — | — | 24.91 | 20.60 | 35.05% |
| `tahrirchi-bs512-lr1e-4` | `train` |  `tahrirchi/tahrirchi-bert-base` | 512 | 1e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.13655 | 0.68450 | 0.68767 | — | — | — | — | 40.00 | 36.40 | 13.52% |
| `uztext-bs1024-lr4e-4` | `train` |  `rifkat/uztext-3Gb-BPE-Roberta` | 1024 | 4e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | **0.16225** | **0.62739** | **0.62672** | — | — | — | — | 20.66 | 17.53 | 15.75% |
| `m-distilbert-bs1024-lr4e-4` | `train` |  `distilbert/distilbert-base-multilingual-cased` | 1024 | 4e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.13874 | 0.60901 | 0.61303 | — | — | — | — | 25.12 | 20.66 | 34.95% |
| `bertbek-sft-bs512-lr2e-4` | `train` |  `elmurod1202/bertbek-ner-uznews` | 512 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | **0.20376** | **0.59090** | **0.59042** | — | — | — | — | 42.23 | 38.44 | 14.17% |
| `m-distilbert-bs1024-lr1e-4` | `train` |  `distilbert/distilbert-base-multilingual-cased` | 1024 | 1e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.15027 | 0.58920 | 0.59492 | — | — | — | — | 25.45 | 20.63 | 34.99% |
| `uztext-bs1024-lr2e-4` | `train` |  `rifkat/uztext-3Gb-BPE-Roberta` | 1024 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.17614 | 0.58318 | 0.58332 | — | — | — | — | 20.72 | 17.54 | 15.74% |
| `bertbek-sft-bs512-lr5e-5` | `train` |  `elmurod1202/bertbek-ner-uznews` | 512 | 5e-5 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.24041 | 0.51053 | 0.51019 | — | — | — | — | 42.25 | 38.41 | 14.18% |
| `uztext-bs1024-lr8e-4` | `train` |  `rifkat/uztext-3Gb-BPE-Roberta` | 1024 | 8e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | dynamic padding | 2 / 1 | 0.52630 | 0.00000 | 0.00000 | — | — | — | — | 19.64 | 17.50 | 15.77% |
| `large-bs192-lr2e-4-fixed` | `train` |  `FacebookAI/xlm-roberta-large` | 192 | 2e-4 | 3 | SDPA, FP32 weights + BF16 AMP | fused | fixed padding 256 | 2 / 1 | 0.07018 | — | — | — | — | — | — | 117.72 | 103.97 | 23.81% |

Метрики по алфавиту рассчитаны по доминирующему алфавиту документа; смешанные
документы отнесены к латинице или кириллице по большинству букв. Три документа
с равным количеством или без таких букв не включены в эти четыре колонки.

Лучший сохранённый результат по качеству — `xl-bs12-ga3-ebs36-lr5e-5`; он улучшил
micro-F1 относительно `large-bs192-lr2e-4-w2p1` с 0.85450 до 0.86020.
Sortish batching был значительно быстрее, но менял порядок и token composition
батчей и снижал exact-span F1. FlashAttention varlen повышал throughput, но в
проверенных конфигурациях также не превзошёл SDPA по F1.

Для XL в проверенном трехэпоховом режиме gradient accumulation улучшил качество:
effective BS36 (`BS12 × GA3`) превзошёл effective BS28/32. Effective BS48 уже
снизил качество из-за меньшего числа optimizer updates на эпоху. Внутри
effective BS32 локальный optimum LR — `5e-5`; `4e-5` и `7e-5` дали худший F1.

### Fine-tune `FacebookAI/xlm-roberta-large` на `train_v2`

Исторический лучший checkpoint `large-bs192-lr2e-4-w2p1` ранее был очищен из
`artifacts/`, поэтому он был заново воспроизведён с теми же параметрами. Loss по
всем трём эпохам совпал с историческим run, что подтвердило идентичное
восстановление весов. Последующий fine-tune на 44 598 документах `train_v2` с
`BS192`, `LR=2e-5`, тремя эпохами снизил dev micro-F1 с 0.85450 до 0.79518.
`train_v2` сильнее смещён к латинице и GEO, чем исходный train/dev; подробная
статистика находится в `EDA_STATS.md`.

Первый LR probe, один полный проход по `train_v2` при `LR=1e-6`, также снизил
micro-F1 до 0.81325. `LR=1e-7` оказался значительно лучше (0.85047), но всё ещё
не превзошёл исходный checkpoint; дальнейший поиск отталкивается от этого узкого
низкого-LR диапазона, кроме отдельно запрошенных диагностических точек.

### Loss по эпохам

| Run | Epoch 1 train/dev | Epoch 2 train/dev | Epoch 3 train/dev |
|---|---:|---:|---:|
| `large-bs192-lr2e-4-w2p1` |  0.31515 / 0.11980 | 0.06352 / 0.07387 | 0.03321 / 0.07018 |
| `large-bs192-lr2e-4-fixed` |  0.31515 / 0.11980 | 0.06352 / 0.07387 | 0.03321 / 0.07018 |
| `large-bs192-lr2e-4-sortish20` |  0.53877 / 0.14633 | 0.09323 / 0.08796 | 0.05507 / 0.06840 |
| `large-bs192-lr2e-4-sortish5` |  0.50421 / 0.17254 | 0.11779 / 0.08554 | 0.05565 / 0.07158 |
| `large-bs192-lr2e-4-fa-varlen-unfused` |  0.27298 / 0.10464 | 0.06575 / 0.07638 | 0.03682 / 0.07142 |
| `large-bs192-lr2e-4-fa-varlen` |  0.38927 / 0.09155 | 0.05695 / 0.06447 | 0.03682 / 0.06546 |
| `large-bs192-lr1.5e-4-fa-varlen` |  0.39653 / 0.07439 | 0.05537 / 0.06342 | 0.03854 / 0.06440 |
| `xl-bs28-lr2e-5` |  0.53206 / 0.16107 | 0.11153 / 0.09873 | 0.07213 / 0.08878 |
| `xl-bs28-lr5e-5` |  0.37404 / 0.08790 | 0.05799 / 0.06693 | 0.03082 / 0.06507 |
| `xl-bs16-ga2-ebs32-lr4e-5` |  0.43561 / 0.11087 | 0.07408 / 0.07354 | 0.04196 / 0.06954 |
| `xl-bs16-ga2-ebs32-lr5e-5` |  0.40435 / 0.10187 | 0.06576 / 0.07050 | 0.03613 / 0.06750 |
| `xl-bs16-ga2-ebs32-lr7e-5` |  0.35902 / 0.08743 | 0.05455 / 0.06496 | 0.02711 / 0.06587 |
| `xl-bs12-ga3-ebs36-lr5e-5` |  0.41258 / 0.10172 | 0.06813 / 0.07081 | 0.03798 / 0.06758 |
| `xl-bs12-ga4-ebs48-lr5e-5` |  0.48513 / 0.12941 | 0.08667 / 0.08259 | 0.05073 / 0.07619 |
| `large-best-to-train-v2-bs192-lr2e-5` |  0.13441 / 0.07887 | 0.08991 / 0.07908 | 0.07651 / 0.08306 |
| `large-best-to-v2-bs192-lr1e-6-e1` |  0.19104 / 0.07325 | — | — |
| `large-best-to-v2-bs192-lr1e-7-e1` |  0.28007 / 0.06568 | — | — |
| `large-best-to-v2-bs192-lr1.5e-4-e1` | 0.11436 / 0.07496 | — | — |
| `large-best-to-v2-bs192-lr3e-8-e1` | 0.30026 / 0.06838 | — | — |
| `large-best-to-v2-bs192-lr5e-8-e1` | 0.29425 / 0.06738 | — | — |
| `large-best-to-v2-bs192-lr3e-8-e2` | 0.29914 / 0.06767 | 0.28345 / 0.06692 | — |
| `large-best-to-v2-bs192-lr3e-8-e3` | 0.29954 / 0.06757 | 0.27897 / 0.06623 | 0.26932 / 0.06593 |
| `large-best-to-v2-bs128-lr3e-8-e1` | 0.29665 / 0.06770 | — | — |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-6-e1` | 0.15992 / 0.08248 | — | — |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-8-e1` | 0.25241 / 0.06753 | — | — |
| `xl-best-to-v3-bs12-ga3-ebs36-lr1e-8-e1` | 0.25113 / 0.06755 | — | — |
| `large-best-to-v2-bs256-lr3e-8-e1` | 0.30256 / 0.06878 | — | — |
| `large-best-to-v2-bs160-ga2-ebs320-lr3e-8-e1` | 0.30406 / 0.06897 | — | — |
| `tahrirchi-bs512-lr1e-4` |  0.49410 / 0.17629 | 0.14059 / 0.14152 | 0.10688 / 0.13655 |
| `tahrirchi-bs512-lr2e-4` |  0.46384 / 0.20938 | 0.13171 / 0.13829 | 0.08948 / 0.13074 |
| `tahrirchi-bs512-lr2.5e-4` |  0.49631 / 0.19493 | 0.13202 / 0.13908 | 0.08676 / 0.13498 |
| `tahrirchi-bs512-lr3e-4` |  0.48642 / 0.22107 | 0.13508 / 0.13847 | 0.08631 / 0.13310 |
| `tahrirchi-bs512-lr2.5e-4-e5` |  0.46523 / 0.20534 | 0.13104 / 0.13297 | 0.08133 / 0.13378 (epoch 4: 0.05406 / 0.13829; epoch 5: 0.03849 / 0.14919) |
| `uztext-bs1024-lr2e-4` |  0.77676 / 0.28530 | 0.22589 / 0.19024 | 0.16932 / 0.17614 |
| `uztext-bs1024-lr4e-4` |  0.93978 / 0.24156 | 0.19236 / 0.17164 | 0.13773 / 0.16225 |
| `uztext-bs1024-lr8e-4` |  1.28296 / 0.52630 | 0.49767 / 0.53790 | 0.52137 / 0.55644 |
| `m-distilbert-bs1024-lr1e-4` |  0.71160 / 0.30885 | 0.21596 / 0.17233 | 0.14736 / 0.15027 |
| `m-distilbert-bs1024-lr2e-4` |  0.60843 / 0.21862 | 0.16058 / 0.13383 | 0.10962 / 0.11970 |
| `m-distilbert-bs1024-lr4e-4` |  0.66351 / 0.36357 | 0.23285 / 0.17496 | 0.13435 / 0.13874 |
| `bertbek-sft-bs512-lr5e-5` |  0.40597 / 0.27910 | 0.24307 / 0.25646 | 0.21566 / 0.24041 |
| `bertbek-sft-bs512-lr2e-4` |  0.35821 / 0.25587 | 0.19867 / 0.20757 | 0.16027 / 0.20376 |

### Детальные метрики новых моделей

В ячейках классов указан формат `Precision / Recall / F1`.

| Run | ORG | NAME | GEO | Micro P/R/F1 | Macro P/R/F1 |
|---|---:|---:|---:|---:|---:|
| `xl-bs28-lr5e-5` |  .7966/.8484/.8217 | .8596/.8952/.8771 | .8649/.8801/.8724 | .8392/.8737/.8561 | .8404/.8746/.8571 |
| `xl-bs16-ga2-ebs32-lr4e-5` |  .7987/.8492/.8232 | .8523/.8883/.8699 | .8602/.8706/.8653 | .8361/.8685/.8520 | .8371/.8694/.8528 |
| `xl-bs16-ga2-ebs32-lr5e-5` |  .8124/.8552/.8333 | .8650/.8952/.8798 | .8559/.8710/.8633 | .8434/.8728/.8578 | .8444/.8738/.8588 |
| `xl-bs16-ga2-ebs32-lr7e-5` |  .8195/.8349/.8271 | .8580/.8780/.8679 | .8534/.8882/.8705 | .8432/.8667/.8548 | .8436/.8670/.8552 |
| `xl-bs12-ga3-ebs36-lr5e-5` |  .8089/.8627/.8349 | .8642/.8918/.8778 | .8705/.8702/.8704 | .8466/.8741/.8602 | .8479/.8749/.8610 |
| `xl-bs12-ga4-ebs48-lr5e-5` |  .7942/.8417/.8172 | .8481/.8836/.8655 | .8580/.8618/.8599 | .8324/.8614/.8467 | .8334/.8623/.8475 |
| `large-best-to-train-v2-bs192-lr2e-5` |  .7499/.7995/.7739 | .8318/.8232/.8275 | .7674/.8129/.7895 | .7797/.8114/.7952 | .7831/.8119/.7970 |
| `large-best-to-v2-bs192-lr1e-6-e1` |  .7889/.7732/.7810 | .8026/.8189/.8107 | .8168/.8768/.8457 | .8033/.8236/.8133 | .8028/.8230/.8125 |
| `large-best-to-v2-bs192-lr1e-7-e1` |  .8068/.8372/.8217 | .8575/.8693/.8634 | .8564/.8794/.8678 | .8394/.8618/.8505 | .8402/.8620/.8510 |
| `large-best-to-v2-bs192-lr1.5e-4-e1` | .7712/.8315/.8002 | .8289/.8650/.8466 | .8270/.8787/.8520 | .8080/.8583/.8324 | .8090/.8584/.8330 |
| `large-best-to-v2-bs192-lr3e-8-e1` | .8000/.8514/.8249 | .8616/.8775/.8695 | .8582/.8875/.8726 | .8386/.8720/.8550 | .8399/.8722/.8557 |
| `large-best-to-v2-bs192-lr5e-8-e1` | .8024/.8477/.8244 | .8604/.8745/.8674 | .8594/.8857/.8724 | .8396/.8692/.8542 | .8408/.8693/.8547 |
| `large-best-to-v2-bs192-lr3e-8-e2` | .8059/.8466/.8258 | .8616/.8749/.8682 | .8606/.8849/.8726 | .8417/.8687/.8550 | .8427/.8688/.8555 |
| `large-best-to-v2-bs192-lr3e-8-e3` | .8056/.8402/.8225 | .8561/.8698/.8629 | .8573/.8812/.8691 | .8389/.8636/.8511 | .8397/.8637/.8515 |
| `large-best-to-v2-bs128-lr3e-8-e1` | .8009/.8488/.8242 | .8613/.8754/.8683 | .8586/.8860/.8721 | .8390/.8700/.8542 | .8403/.8701/.8548 |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-6-e1` | .7474/.7067/.7265 | .7758/.7891/.7824 | .6688/.7210/.6939 | .7265/.7366/.7315 | .7307/.7389/.7343 |
| `xl-best-to-v2-bs12-ga3-ebs36-lr1e-8-e1` | .8154/.8571/.8357 | .8692/.8823/.8757 | .8725/.8680/.8703 | .8512/.8685/.8598 | .8523/.8691/.8605 |
| `xl-best-to-v3-bs12-ga3-ebs36-lr1e-8-e1` | .8148/.8571/.8354 | .8695/.8823/.8759 | .8721/.8676/.8699 | .8509/.8684/.8596 | .8522/.8690/.8604 |
| `large-best-to-v2-bs256-lr3e-8-e1` | .7992/.8518/.8247 | .8614/.8788/.8700 | .8588/.8879/.8731 | .8385/.8727/.8553 | .8398/.8728/.8559 |
| `large-best-to-v2-bs160-ga2-ebs320-lr3e-8-e1` | .7992/.8533/.8254 | .8603/.8793/.8697 | .8583/.8882/.8730 | .8380/.8735/.8554 | .8393/.8736/.8560 |
| `tahrirchi-bs512-lr1e-4` |  .5365/.6721/.5967 | .6733/.7766/.7213 | .7180/.7743/.7451 | .6370/.7397/.6845 | .6426/.7410/.6877 |
| `tahrirchi-bs512-lr2e-4` |  .5838/.7386/.6522 | .6939/.7809/.7348 | .7525/.7926/.7721 | .6710/.7705/.7173 | .6767/.7707/.7197 |
| `tahrirchi-bs512-lr2.5e-4` |  .5867/.7281/.6498 | .7047/.7883/.7441 | .7596/.7875/.7733 | .6778/.7672/.7197 | .6836/.7680/.7224 |
| `tahrirchi-bs512-lr3e-4` |  .5815/.7315/.6479 | .6896/.7693/.7273 | .7609/.7886/.7745 | .6712/.7631/.7142 | .6773/.7631/.7166 |
| `tahrirchi-bs512-lr2.5e-4-e5` |  .5644/.7067/.6276 | .6753/.7831/.7252 | .7546/.7618/.7581 | .6580/.7492/.7006 | .6648/.7505/.7037 |
| `uztext-bs1024-lr2e-4` |  .4622/.5848/.5164 | .4875/.6235/.5472 | .6523/.7243/.6864 | .5317/.6458/.5832 | .5340/.6442/.5833 |
| `uztext-bs1024-lr4e-4` |  .5290/.6277/.5741 | .5160/.6658/.5814 | .6933/.7588/.7246 | .5784/.6855/.6274 | .5795/.6841/.6267 |
| `uztext-bs1024-lr8e-4` |  0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| `m-distilbert-bs1024-lr1e-4` |  .4421/.5182/.4771 | .6629/.7421/.7003 | .5912/.6243/.6073 | .5588/.6231/.5892 | .5654/.6282/.5949 |
| `m-distilbert-bs1024-lr2e-4` |  .5605/.6337/.5949 | .7247/.7934/.7575 | .6945/.7371/.7152 | .6558/.7184/.6857 | .6599/.7214/.6892 |
| `m-distilbert-bs1024-lr4e-4` |  .4893/.5773/.5297 | .6536/.7210/.6857 | .5838/.6695/.6237 | .5705/.6532/.6090 | .5756/.6559/.6130 |
| `bertbek-sft-bs512-lr5e-5` |  .4174/.5299/.4669 | .4451/.6244/.5197 | .4759/.6346/.5439 | .4469/.5953/.5105 | .4461/.5963/.5102 |
| `bertbek-sft-bs512-lr2e-4` |  .5038/.6420/.5646 | .5084/.6628/.5754 | .5792/.6937/.6313 | .5307/.6665/.5909 | .5305/.6662/.5904 |

## Zero-shot `bertbek-ner-uznews`

Метки исходной модели нормализованы как `PERSON -> NAME`, `LOCATION -> GEO`,
`ORG -> ORG`; `DATE`, `MISC` и `TIME` считаются `O`. На полной dev-выборке:

- ORG P/R/F1: 0.1438 / 0.1023 / 0.1196;
- NAME P/R/F1: 0.1143 / 0.1574 / 0.1324;
- GEO P/R/F1: 0.2837 / 0.2886 / 0.2861;
- micro P/R/F1: 0.1811 / 0.1847 / 0.1829;
- macro P/R/F1: 0.1806 / 0.1828 / 0.1794;
- end-to-end inference: 12.516 s for 1 500 documents / 2 159 windows
  (119.8 documents/s, including model load and tokenization; batch size 256).

Для SFT исходные строки classifier для `ORG`, `PERSON`, `LOCATION` и `O` были
перенесены в семиклассовую схему кейса, то есть pretrained NER head не
переинициализировалась. Лучший из двух SFT запусков (`lr=2e-4`) дал micro-F1
0.5909; end-to-end inference занял 12.925 s (116.1 documents/s).

## Одноэпоховые и диагностические прогоны

Эти результаты нельзя напрямую сравнивать с полными экспериментами.

| Run | Model / data | BS | LR | Attention / weights | AdamW | Train/dev loss | Time, s | Compute, s | HFU | Result |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| `large-fa-varlen-unfused-e1` |  large, full data | 192 | 2e-4 | FA2, FP32 + AMP | unfused | 0.26691 / 0.08033 | 37.90 | 33.42 | 24.70% | completed, metrics not calculated |
| `large-fa-varlen-bf16-fused-e1` |  large, full data | 192 | 2e-4 | FA2, direct BF16 | fused | 0.27113 / 0.07693 | 34.99 | 32.00 | 25.79% | completed, metrics not calculated |
| `xl-sdpa-bs16-probe` |  XL, 192/64 documents | 16 | 2e-5 | SDPA, FP32 + AMP | fused | 1.83939 / 1.72090 | 23.54 | 4.59 | 10.67% | capacity probe only |
| `xl-sdpa-bs24-probe` |  XL, 192/64 documents | 24 | 2e-5 | SDPA, FP32 + AMP | fused | 1.88758 / 1.82079 | — | — | — | trained; save failed: no disk space |
| `xl-fa-bs32-probe` |  XL, 192/64 documents | 32 | 5e-5 | FA2, direct BF16 | fused | 1.86000 / 1.83839 | — | — | 14.79% | capacity probe only |
| `xl-fa-bs64-probe` |  XL, 256/64 documents | 64 | 5e-5 | FA2, direct BF16 | fused | 1.86974 / 1.84715 | — | — | 18.24% | capacity probe only |

## Неуспешные и прерванные эксперименты

| Run/configuration | Result |
|---|---|
| XL FA2, BS128 | OOM on first step at about 79.08 GiB |
| XL FA2, BS112 | OOM on first step at about 79.06 GiB |
| XL FA2, BS80 | OOM after first optimizer step at about 78.93 GiB |
| XL FA2, BS64 on full dataset | OOM on first long batch at about 79.08 GiB |
| XL FA2, BS48, LR 2e-5 | stopped in epoch 2; epoch 1 train/dev loss 1.05665 / 0.65579, unhealthy optimization |
| XL SDPA, BS20, LR 2e-5 | stable at about 3.42 batches/s; deliberately stopped at step 108/923 to test BS28 |
| XL SDPA, BS28 × GA2 (effective BS56), LR 5e-5 | OOM at first optimizer step |
| XL SDPA, BS24 × GA2 (effective BS48), LR 5e-5 | OOM at first optimizer step |
| Large `torch.compile` | failed because Triton/CUDA toolkit 13.1 did not match PyTorch CUDA 12.4 |
| Large FA2 with fused AdamW and FP32 master weights | did not converge; fixed operationally by direct BF16 weights, which reduced F1 |

## Исторические конфигурации без сохранённых результатов

Для следующих каталогов результаты были удалены при очистке воспроизводимых
checkpoint-ов и сейчас не могут быть достоверно внесены в метрики. Имена
сохраняются как журнал того, что конфигурации запускались или планировались.

| Artifact directory | Recoverable configuration |
|---|---|
| `base-facebookAI-xlm-roberta-large-bf16-b128-lr1e-4` |  large, BF16, BS128, LR 1e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4` |  large, BF16, BS192, LR 2e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-bucket` |  large, BF16, BS192, LR 2e-4, naive length buckets |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-fa` |  large, BF16, BS192, LR 2e-4, earlier SDPA/Flash kernel experiment |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-fused` |  large, BF16, BS192, LR 2e-4, fused AdamW |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-hfu` |  large, BF16, BS192, LR 2e-4, HFU instrumentation baseline |
| `base-facebookAI-xlm-roberta-large-bf16-bs192-lr2e-4-opt1` |  large, BF16, BS192, LR 2e-4, first throughput optimizations |
| `base-facebookAI-xlm-roberta-large-bf16-bs224-lr1e-4` |  large, BF16, BS224, LR 1e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs224-lr2e-4` |  large, BF16, BS224, LR 2e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs224-lr2.5e-4` |  large, BF16, BS224, LR 2.5e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs224-lr2.5e-4-epochs5` |  large, BF16, BS224, LR 2.5e-4, 5 epochs |
| `base-facebookAI-xlm-roberta-large-bf16-bs224-lr3e-4` |  large, BF16, BS224, LR 3e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs240-lr2.5e-4` |  large, BF16, BS240, LR 2.5e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs256-lr2e-4` |  large, BF16, BS256, LR 2e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs64-lr1e-4` |  large, BF16, BS64, LR 1e-4 |
| `base-facebookAI-xlm-roberta-large-bf16-bs64-lr5e-5` |  large, BF16, BS64, LR 5e-5 |
| `base-facebookAI-xlm-roberta-large-bf16-bs8-lr5e-5` |  large, BF16, BS8, LR 5e-5 |
| `tune-xlmr-large-bf16-b4-lr2e-5` |  large, BF16, BS4, LR 2e-5 |
| `tune-xlmr-large-bf16-bs128-lr2e-4` |  large, BF16, BS128, LR 2e-4 |
| `tune-xlmr-large-bf16-bs32-lr5e-5-e5` |  large, BF16, BS32, LR 5e-5, 5 epochs |
| `tune-xlmr-large-bf16-bs64-lr3e-5-e5` |  large, BF16, BS64, LR 3e-5, 5 epochs |

## Исправление координат для XLM-R XL

Первоначальная оценка XL ошибочно дала micro-F1 0.15365. Причиной был tokenizer
backend модели `facebook/xlm-roberta-xl`: он включал разделяющий пробел в offset
следующего токена, например `(17, 25)` для слова, реально расположенного в
`(18, 25)`. После удаления ведущего и завершающего whitespace из token offsets
получены корректные micro-F1 0.82219 и macro-F1 0.82351 без переобучения.

## Как обновлять таблицу

После каждого полного эксперимента необходимо сохранить:

1. `training_summary.json` из каталога запуска;
2. `dev_predictions.jsonl`;
3. `dev_metrics.json`, рассчитанный `scripts/evaluate.py`;
4. строку в основной таблице этого файла, включая не только F1, но и loss,
   wall/compute time, HFU и все параметры, отличающиеся от общих условий.

## Additional experiments 2026-09-03

| Run | Configuration | Micro-F1 | Result |
|---|---|---:|---|
| logit-xl | Existing XL checkpoint; overlapping-window logit averaging | 0.86040 | Small improvement over probability averaging |
| xl-original-script-balanced-1e | XL, original train with Cyrillic/mixed oversampling, 1 epoch, BS12/GA3, LR 5e-5 | 0.77370 | Rejected; duplication destabilized the classifier |
| ensemble-xl-large-0.75 | XL checkpoint + saved XLM-R large checkpoint; averaged logits, weights 1.0/0.75 | 0.87330 | Current best |
| ensemble-xl-tuned-large | XL checkpoint + tuned large checkpoint; averaged logits, weights 1.0/0.75 | 0.87240 | Rejected; slightly below current best |
| gazetteer-m2-p100 | Current-best ensemble plus exact training-surface gazetteer | 0.36780 | Rejected; unacceptable false-positive rate |

Current-best inference uses scripts/ensemble_predict.py with the XL checkpoint weight 1.0 and the saved large checkpoint weight 0.75, followed by scripts/evaluate.py.

## Active learning and constrained decoding follow-up

| Run | Configuration | Micro-F1 | Result |
|---|---|---:|---|
| ensemble-xl-large-0.75-constrained | Existing XL + large logit ensemble, legal BIO Viterbi decoding | 0.8868 | New best; kept |
| xl-active-filtered250 | Original train minus top 250 train examples ranked by model disagreement/error, XL, 3 epochs, BS12/GA3, LR 5e-5 | 0.8434 | Rejected standalone |
| ensemble-active-filtered250 | Existing ensemble plus filtered-data XL checkpoint at weight 0.5, constrained | 0.8862 | Rejected; below constrained best |
| calibration-global-minus0.15 | Existing ensemble, constrained, -0.15 bias on all entity labels | 0.8865 | Rejected |
| calibration-global-plus0.15 | Existing ensemble, constrained, +0.15 bias on all entity labels | 0.8866 | Rejected |

The active-learning report (`scripts/mine_active_learning.py`) identified a
large concentration of long news, advertising, foreign-language and duplicate
stories among high-disagreement training examples. Removing them as a single
batch did not improve generalization, so the filtered dataset is not part of
the production recipe. The report and filter remain available for manual
review rather than automatic deletion.

## Top-1000 candidate review

The top 1000 training candidates were reviewed with
`scripts/review_active_learning.py`. The policy added only exact spans shared
by both prediction views when they did not overlap an existing annotation.
Existing annotations were never automatically deleted or boundary-adjusted;
gold spans missed by both models were logged for possible human review.

| Artifact | Count/result |
|---|---:|
| Candidate records reviewed | 1000 |
| Records changed | 315 |
| Exact consensus spans added | 799 |
| Overlapping consensus spans skipped | 457 |
| Gold spans missed by both, retained | 985 |
| Reviewed XL checkpoint standalone micro-F1 | 0.7265 |
| Reviewed checkpoint in constrained ensemble, weight 0.25 | 0.8863 |

The reviewed dataset was rejected for production because 0.8863 is below the
0.8868 constrained baseline. The new dataset and per-record audit remain
available as `data/train_active_reviewed_consensus_add.jsonl` and
`artifacts/active_learning_train_review_audit.jsonl`.

## Genuine manual semantic review

The previous consensus pass is explicitly not counted as human review. A
persistent manual ledger was added with `scripts/export_manual_review_batch.py`
and `scripts/apply_manual_review.py`. Batch 001 manually inspected two ranked
candidates. The Spanish football roster was corrected by replacing malformed
fragment predictions and incomplete labels with complete person-name spans.
The keyword/search-list record was marked `review_required` and left
unchanged because its annotation policy is ambiguous. The full 1000-record
manual review remains in progress; no automatic consensus labels are included
in this manual batch.

Manual batch 001 was extended to four explicit decisions: one roster
replacement, two scraped keyword/social-tag records excluded from training, and
one keyword-list record retained as `review_required`. The materialized dataset
contains 12,998 records; the original 13,000-record training file is untouched.
After the next four ranked candidates were inspected, one duplicated bilingual
calendar scrape was also excluded and two long editorial/scraped documents
were marked `review_required`. The current manually decided output therefore
contains 12,997 records; only explicit ledger actions are applied.
