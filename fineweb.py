"""
Build FineWeb-Edu token shards for SRS pretraining.

Source dataset:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

This script downloads the dataset, tokenizes it with GPT-2 BPE (tiktoken),
and writes fixed-size token shards to disk.

Usage:
    $ python build_fineweb_edu_shards.py

Output:
    Shards are saved under the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
from typing import Iterable

import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# --------------------------- Configuration --------------------------- #
OUTPUT_DIR_NAME = "edu_fineweb10B"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG_NAME = "sample-10BT"  # HF "name" for this dataset
SHARD_SIZE_TOKENS = int(1e8)  # 100M tokens per shard

# ----------------------------- Paths -------------------------------- #
SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Tokenization --------------------------- #
_tokenizer = tiktoken.get_encoding("gpt2")
_EOT = _tokenizer._special_tokens["<|endoftext|>"]  # end-of-text token id


def tokenize(example: dict) -> np.ndarray:
    """
    Tokenize a single dataset example to uint16 token ids.

    The EOT token is prepended to delimit documents.
    """
    tokens = [_EOT]
    # encode_ordinary avoids adding any special tokens or BOS/EOS
    tokens.extend(_tokenizer.encode_ordinary(example["text"]))
    tokens_np = np.array(tokens, dtype=np.int64)

    # Safety: ensure ids fit into uint16
    if not ((tokens_np >= 0).all() and (tokens_np < 2**16).all()):
        raise ValueError("Token id range exceeds uint16; choose a wider dtype.")

    return tokens_np.astype(np.uint16)


def save_shard(filepath: str, tokens: np.ndarray) -> None:
    """Save a token shard as a .npy file."""
    np.save(filepath, tokens)


def iter_tokenized(
    dataset_iter: Iterable[dict], processes: int
) -> Iterable[np.ndarray]:
    """Tokenize dataset examples in parallel and yield token arrays."""
    # chunksize affects scheduling; 16 is a good default for medium docs
    with mp.Pool(processes) as pool:
        for tok in pool.imap(tokenize, dataset_iter, chunksize=16):
            yield tok


def build_shards() -> None:
    # Download/stream dataset (train split)
    ds = load_dataset(DATASET_ID, name=DATASET_CONFIG_NAME, split="train")

    # Parallelism: conservative default for stability
    num_procs = max(1, os.cpu_count() // 2)

    shard_idx = 0
    shard_buf = np.empty((SHARD_SIZE_TOKENS,), dtype=np.uint16)
    fill = 0
    pbar = None

    for tokens in iter_tokenized(ds, processes=num_procs):
        remaining = SHARD_SIZE_TOKENS - fill

        # If the whole token array fits, copy and continue
        if len(tokens) <= remaining:
            shard_buf[fill : fill + len(tokens)] = tokens
            fill += len(tokens)

            if pbar is None:
                pbar = tqdm(total=SHARD_SIZE_TOKENS, unit="tokens", desc=f"Shard {shard_idx}")
            pbar.update(len(tokens))
            continue

        # Otherwise, fill this shard to capacity, save, and carry remainder
        if pbar is None:
            pbar = tqdm(total=SHARD_SIZE_TOKENS, unit="tokens", desc=f"Shard {shard_idx}")
        pbar.update(remaining)

        shard_buf[fill : fill + remaining] = tokens[:remaining]
        split = "val" if shard_idx == 0 else "train"
        shard_path = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}.npy")
        save_shard(shard_path, shard_buf)
        pbar.close()
        pbar = None

        shard_idx += 1

        # Start next shard with leftovers
        leftovers = tokens[remaining:]
        fill = len(leftovers)
        shard_buf[:fill] = leftovers

    # Flush final partial shard, if any
    if fill > 0:
        split = "val" if shard_idx == 0 else "train"
        shard_path = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}.npy")
        save_shard(shard_path, shard_buf[:fill])
        if pbar is not None:
            pbar.close()


def main() -> None:
    build_shards()


if __name__ == "__main__":
    # Guard needed for Windows multiprocessing
    main()
