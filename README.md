# Mini-GPT — From-Scratch Transformer Language Model

## Overview
This repo contains a from-scratch, mini-scale GPT-style language model in PyTorch. It mirrors a GPT-2-like architecture (124M by default) and includes a clean training loop, optional distributed training, and simple evaluation/generation utilities.

This is a pure language model (no chat finetuning). Outputs are raw continuations of the training distribution.

## What’s in the box
- **Modular Transformer**: Causal self-attention blocks (multi-head) + MLP with GELU, RMSInit-style scaling on select projections, and weight tying (`wte` ↔ `lm_head`).
- **Cohesive naming**: `snake_case` across variables & functions, clear module boundaries.
- **Tokenization**: GPT-2 BPE via `tiktoken` (with a tiny char fallback in utils if you keep it).
- **Training**: DDP-ready, AMP (`bfloat16`) autocast, cosine LR schedule with warmup, grad accumulation, AdamW (fused on CUDA when available).
- **Data**: Works with Tiny Shakespeare or large corpora via optional **FineWeb-Edu** shard builder.
- **Evaluation**: HellaSwag accuracy (length-normalized completion loss) and perplexity; text generation with top-k sampling.

## Repo layout (recommended)
```
.
├── gpt_model.py                  # GPT, Block, MLP, CausalSelfAttention, GPTConfig
├── train_gpt.py                  # Training script (DDP/AMP/grad-accum/logging)
├── eval_hellaswag.py             # HellaSwag eval (refactored names)
├── build_fineweb_edu_shards.py   # Optional: download/tokenize FineWeb-Edu into .npy shards
├── utils/
│   ├── data_loader.py            # (If kept) helpers & char-tokenizer fallback
│   └── …
├── hellaswag/                    # auto-downloaded JSONL files (val/train/test)
├── edu_fineweb10B/               # generated token shards (.npy), if you use FineWeb-Edu
└── README.md
```

> If your filenames differ, keep them—just ensure imports match the refactored names below.

## Requirements
- Python 3.8+
- PyTorch 2.0+
- `tiktoken`, `numpy`, `tqdm`, `matplotlib`, `datasets` (if using FineWeb-Edu)

```bash
pip install torch tiktoken numpy tqdm matplotlib datasets
```

## Datasets

### Option A: Tiny Shakespeare (quick start)
Place `input.txt` in the repo root (unchanged).

### Option B (optional): FineWeb-Edu shards
Download & tokenize **FineWeb-Edu**; write fixed-size `.npy` shards:

```bash
python build_fineweb_edu_shards.py
# outputs: ./edu_fineweb10B/edufineweb_{val|train}_000000.npy, ...
```

## Usage

### 1) Train
Single GPU / CPU:
```bash
python train_gpt.py
```

Multi-GPU (DDP):
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Adjust near the top of `train_gpt.py`:
- `TOTAL_BATCH_TOKENS`, `MICRO_BATCH`, `SEQ_LEN`
- LR schedule: `MAX_LR`, `MIN_LR`, `WARMUP_STEPS`, `MAX_STEPS`
- Data source: shards folder or Tiny Shakespeare

### 2) Generate
Generation is baked into the training loop (periodically) and can be called ad-hoc. Example prompt:
```
"Hello, I'm a language model,"
```
(Optionally add a small `generate.py` if you prefer a separate entrypoint.)

### 3) Evaluate on HellaSwag
```bash
python eval_hellaswag.py -m gpt2 -d cuda
```
Uses `render_multiple_choice_tensors` + `iter_examples`. Prints running `acc_norm` and a final summary.

## Model (`gpt_model.py`)
- **Classes**
  - `GPTConfig`: `block_size=1024`, `vocab_size=50257` (or 50304 if padded), `n_layer=12`, `n_head=12`, `n_embd=768`
  - `CausalSelfAttention`, `MLP`, `Block`, `GPT`
- **Init**
  - Linear/Embedding weights ~N(0, std) with scaled init on projection layers:  
    `std *= (2 * n_layer) ** -0.5` when `NANOGPT_SCALE_INIT` is set.
- **Forward**
  - Token + position embeddings → blocks → `ln_f` → `lm_head`.  
  - Returns `(logits, loss)`; CE loss if `targets` provided.
- **Pretrained import**
  - `GPT.from_pretrained("gpt2|gpt2-medium|gpt2-large|gpt2-xl")` maps HF weights (transposes Conv1D weights where needed).

## Data loader
- **Tiny**: reads `input.txt`, encodes with `tiktoken` (or char fallback).
- **FineWeb-Edu**: reads `.npy` shards from `./edu_fineweb10B/` (uint16/32 → LongTensor).
- Batches are (B, T) windows with rank-aware stepping for DDP.

## Training (`train_gpt.py`)
- **DDP**: set up via `torchrun`; env-provided rank/world handled automatically.
- **AMP**: `torch.autocast` with `bfloat16` where supported.
- **Optimizer**: AdamW (fused on CUDA when available).  
  Grouped weight decay: params with `dim>=2` decay; biases/LayerNorms do not.
- **LR schedule**: Warmup → cosine → floor at `MIN_LR`.
- **Logging**: `./log/log.txt` lines like  
  `"{step} train {loss}"`, `"{step} val {loss}"`, `"{step} hella {acc}"`.
- **Checkpoints**: `./log/model_{step}.pt`.

## Evaluation
- **HellaSwag** (`eval_hellaswag.py`):  
  length-normalized completion loss → argmin across 4 options.
- **Perplexity**: compute from validation CE on your corpus.

## Results (example, tiny runs)
- Starting loss ~11 → ~4–5 (tiny corpus).
- See `log/log.txt` for curves; use your notebook for plots.

## Naming & API changes (migration map)
Updating an existing codebase to match this refactor:
- **HellaSwag script**
  - `render_example` → `render_multiple_choice_tensors`
  - `iterate_examples` → `iter_examples`
  - `--model_type` → `--model_name`
- **Training**
  - Global flags upper-cased: `TOTAL_BATCH_TOKENS`, `MICRO_BATCH`, `SEQ_LEN`, …
  - LR fn: `get_lr` → `learning_rate_for`
  - Helper: `get_most_likely_row` → `most_likely_completion_index`
  - Optimizer builder takes `is_master_process` instead of reading a global.
- **FineWeb script**
  - `fineweb.py` → `build_fineweb_edu_shards.py`
  - saves explicit `.npy` shards

## Limitations
- Small(ish) model; large-scale training needs more compute & data.
- Plain LM—no RLHF/Chat/Instruction tuning.
