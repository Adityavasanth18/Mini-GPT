# Mini-GPT
A from-scratch PyTorch transformer language model trainer, built with custom optimizations for efficient text generation on modest hardware.
#!/bin/bash

# Script to add README.md to your GitHub repo
# Run this in your local repo directory after git init or cloning

# Create README.md with the content
cat << 'EOF' > README.md
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
Place text data in `data/input.txt` or let the script download Tiny Shakespeare.

## Usage

1. **Prepare Data**:
