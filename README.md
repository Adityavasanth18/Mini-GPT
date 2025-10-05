# My Mini-GPT: From-Scratch Transformer Language Model Trainer

## Overview

This repository contains my from-scratch implementation of a mini-scale GPT-like language model using PyTorch. I built this step-by-step as a personal project to explore transformer architectures, starting from basic components and working up to a full training pipeline.

The model reproduces a GPT-2-style architecture (124M parameters) and can train on text datasets like Tiny Shakespeare. With more resources, it could scale to larger models. This was developed on modest hardware, taking about an hour for a basic run on a GPU. It's a simple language model focused on text generation—no chat fine-tuning here, so outputs are raw "dreams" of the training data.


I plan to expand this with more features in the future.

For questions or issues, feel free to open a discussion.

## Features

- **Modular Transformer Architecture**: Built with customizable layers for attention, feedforward, and embeddings.
- **Hybrid Tokenization**: Uses GPT-2 BPE by default but includes a char-level fallback for flexibility.
- **Efficient Training**: Supports DDP, mixed precision, and a custom learning rate scheduler with plateau extension.
- **Data Handling**: Auto-downloads and preprocesses datasets, with options for custom text.
- **Evaluation and Generation**: Perplexity metrics and sampling for generated text.
- **Custom Optimizations**: Tuned weight init, numerical stability tweaks, and low-resource modes.

## Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- tiktoken (optional for BPE; fallback available)
- NumPy
- Matplotlib (for plots)

Install with:

pip install torch tiktoken numpy matplotlib


### Dataset
Shakesphere dataset

## Usage

1. **Prepare Data**:
python my_trainer.py --prepare_data_only
Encodes text to binaries for training.

2. **Train**:
python my_trainer.py --batch_size=12 --max_iters=5000 --learning_rate=6e-4 --tiny_run
Use `--tiny_run` for quick CPU tests. Full options with `--help`.

3. **Generate**:
python my_trainer.py --generate --max_new_tokens=500 --start_text="To be or not to be"

4. **Evaluate**:
python my_eval.py --model_path checkpoints/ckpt.pt

6. **Experiments**:
Check `my_experiments.ipynb` for interactive runs and plots.

## Results

From my subsample runs:
- Starting loss ~11.0, down to ~4.5 perplexity.
- See `img/loss_curve.png` for visuals.

### Model (my_model.py)
- **LanguageModelTransformer**: Main nn.Module class.
- Config: Dataclass for params like embed_dim=768, num_heads=12, seq_len=1024, token_vocab_size=50288, dropout_rate=0.1.
- Embeddings: Token and positional with dropout.
- TransformerLayer: AttentionHeads + FeedForwardNet with residuals and norms.
 - AttentionHeads: Causal multi-head with QKV, tril mask, epsilon in scaling for stability.
 - FeedForwardNet: Linear-GELU-Linear.
- Init: Custom std=0.019 based on experiments.
- Forward: Efficient pass to lm_head.

### Data Loader (utils/data_loader.py)
- Memory-mapped binaries for batches.
- Hybrid tokenizer: tiktoken or char-level.
- Preprocess: Remove non-ASCII for better tokens.

### Trainer (my_trainer.py)
- Argparse setup.
- DDP, AMP support.
- Loop: Grad accumulation, AdamW, cosine+plateau scheduler.
- Logging and checkpoints.
- Generation with top-k sampling.

### Evaluation (my_eval.py)
- Perplexity on val set.
- Adaptable for other metrics.

### Development Notes
Commits show progression: embeddings first, then attention, etc. Notebook has ablations.

## Limitations
- Nano-scale only; needs GPU for full runs.
- Basic LM—no advanced fine-tuning.
