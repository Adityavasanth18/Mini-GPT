"""
Evaluate GPT-2 on HellaSwag (validation split).

Dataset / code reference:
https://github.com/rowanz/hellaswag

JSONL example:
{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

Notes:
- The validation set has 10,042 examples.
- We compute two accuracies:
    * acc: argmin over total completion loss
    * acc_norm: argmin over average completion loss (length-normalized)
"""

import os
import json
from typing import Dict, Generator, Iterable, Tuple

import requests
import tiktoken
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

CACHE_SUBDIR = "hellaswag"
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, CACHE_SUBDIR)

HELLASWAG_URLS: Dict[str, str] = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# GPT-2 BPE tokenizer (no special tokens automatically added)
TOKENIZER = tiktoken.get_encoding("gpt2")


# ----------------------------------------------------------------------------- #
# I/O helpers
# ----------------------------------------------------------------------------- #

def download_with_progress(url: str, dest_path: str, chunk_bytes: int = 1024) -> None:
    """Stream a file to disk with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as fout, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_bytes):
            if not chunk:
                continue
            written = fout.write(chunk)
            bar.update(written)


def ensure_split_downloaded(split: str) -> str:
    """Make sure the JSONL for the split exists locally; return its path."""
    if split not in HELLASWAG_URLS:
        raise ValueError(f"Unknown split: {split}. Expected one of {list(HELLASWAG_URLS)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, f"hellaswag_{split}.jsonl")

    if not os.path.exists(jsonl_path):
        url = HELLASWAG_URLS[split]
        print(f"Downloading {split} split from {url} ...")
        download_with_progress(url, jsonl_path)

    return jsonl_path


def iter_examples(split: str) -> Generator[dict, None, None]:
    """Yield raw JSON examples one by one for the given split."""
    jsonl_path = ensure_split_downloaded(split)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ----------------------------------------------------------------------------- #
# Tokenization and tensor rendering
# ----------------------------------------------------------------------------- #

def render_multiple_choice_tensors(example: dict) -> Tuple[dict, torch.Tensor, torch.Tensor, int]:
    """
    From one HellaSwag example, build:
        data:    bookkeeping tokens (ctx + each ending) for reproducibility
        tokens:  (4, T) token ids for [context + ending]
        mask:    (4, T) 1s over completion tokens, 0s over context tokens
        label:   int index of the correct ending (0..3)
    """
    context = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    record = {"label": label, "ctx_tokens": None, "ending_tokens": []}

    # Tokenize context and each ending (prepend a space for GPT-2 BPE behavior)
    ctx_tokens = TOKENIZER.encode(context)
    record["ctx_tokens"] = ctx_tokens

    token_rows = []
    mask_rows = []
    for ending in endings:
        end_tokens = TOKENIZER.encode(" " + ending)
        token_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        record["ending_tokens"].append(end_tokens)

    # Collate to a fixed width tensor
    max_len = max(len(row) for row in token_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, m_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row, dtype=torch.long)
        mask[i, : len(m_row)] = torch.tensor(m_row, dtype=torch.long)

    return record, tokens, mask, label


# ----------------------------------------------------------------------------- #
# Evaluation
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model_name: str = "gpt2", device: str = "cuda") -> None:
    """
    Evaluate a GPT-2 LM on HellaSwag (validation split) in a completion-likelihood setup.
    Prints running acc_norm and a few debug examples.
    """
    torch.set_float32_matmul_precision("high")  # enable TF32 where supported

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    # Optional: enable torch.compile for speed on PyTorch 2.x
    # model = torch.compile(model)

    correct_len_norm = 0
    correct_raw = 0
    total = 0

    for example in iter_examples("val"):
        _, tokens, mask, gold = render_multiple_choice_tensors(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Forward pass (B=4, T)
        logits = model(tokens).logits

        # Next-token losses at every position
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_targets = shift_tokens.view(-1)

        per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        per_row_loss = per_token_loss.view(tokens.size(0), -1)

        # Consider only completion region (where mask == 1); mask must be shifted as well
        shift_mask = mask[..., 1:].contiguous()
        masked_loss = per_row_loss * shift_mask

        # Sum loss (raw) and mean loss (length-normalized) per candidate
        loss_sum = masked_loss.sum(dim=1)
        loss_mean = loss_sum / shift_mask.sum(dim=1)

        pred_raw = int(loss_sum.argmin().item())
        pred_len_norm = int(loss_mean.argmin().item())

        total += 1
        correct_raw += int(pred_raw == gold)
        correct_len_norm += int(pred_len_norm == gold)

        print(f"{total} acc_norm: {correct_len_norm}/{total}={correct_len_norm/total:.4f}")

        # Show a few examples for sanity-check
        if total <= 9:
            print("---")
            print(f"Context:\n{example['ctx']}")
            print("Endings:")
            for i, ending in enumerate(example["endings"]):
                print(f"{i} (loss: {loss_mean[i].item():.4f}) {ending}")
            print(f"predicted: {pred_len_norm}, actual: {gold}")

    # Final summary (mirrors the running printouts)
    if total > 0:
        acc = correct_raw / total
        acc_norm = correct_len_norm / total
        print(f"\nFinal — acc: {acc:.4f}, acc_norm: {acc_norm:.4f} over {total} examples.")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="gpt2", help="HF model name")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="torch device, e.g. 'cuda' or 'cpu'")
    args = parser.parse_args()

    evaluate(model_name=args.model_name, device=args.device)
