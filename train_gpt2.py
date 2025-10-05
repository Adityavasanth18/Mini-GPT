import os
import math
import time
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from hellaswag import render_multiple_choice_tensors, iter_examples  # updated names

# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, batched together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        # project to qkv then split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # GPT-2 vocab (50k merges + 256 bytes + eot)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, "
                f"block size is only {self.config.block_size}"
            )
        # token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos)                             # (T, n_embd)
        tok_emb = self.transformer.wte(idx)                             # (B, T, n_embd)
        x = tok_emb + pos_emb
        # transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # final layernorm + head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name: str):
        """Load GPT-2 weights from Hugging Face into this architecture."""
        assert model_name in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_name}")

        # derive config from model_name
        cfg = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_name]
        cfg["vocab_size"] = 50257
        cfg["block_size"] = 1024

        model = GPT(GPTConfig(**cfg))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # drop buffers

        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()
        hf_keys = [
            k
            for k in hf_sd.keys()
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]

        # Conv1D weights in HF need transpose when mapping to Linear here
        needs_t = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        assert len(hf_keys) == len(sd_keys), f"mismatched keys: {len(hf_keys)} != {len(sd_keys)}"

        for k in hf_keys:
            if any(k.endswith(w) for w in needs_t):
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        is_master_process: bool = True,
    ):
        """Create AdamW with decoupled weight decay, grouping by parameter ndim."""
        # gather trainable params
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        if is_master_process:
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters"
            )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if is_master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# -----------------------------------------------------------------------------


import numpy as np
import tiktoken


def load_tokens(path: str) -> torch.Tensor:
    """Load a shard saved as .npy (uint16/32) and return LongTensor of token ids."""
    arr = np.load(path)
    arr = arr.astype(np.int32)
    return torch.tensor(arr, dtype=torch.long)


class DataLoaderLite:
    """Simple shard-aware dataloader for uint16/32 token shards produced earlier."""

    def __init__(self, batch_size: int, seq_len: int, process_rank: int, world_size: int, split: str):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.world_size = world_size
        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shard_files = sorted(
            os.path.join(data_root, s) for s in os.listdir(data_root) if split in s
        )
        assert len(shard_files) > 0, f"no shards found for split {split}"
        self.shards = shard_files
        if MASTER_PROCESS:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        # advance position for this rank
        self.current_pos += B * T * self.world_size

        # move to next shard if we would overflow
        if self.current_pos + (B * T * self.world_size + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B * T * self.process_rank

        return x, y


# -----------------------------------------------------------------------------


def most_likely_completion_index(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Return argmin completion index using length-normalized loss over completion region."""
    # compute next-token losses
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_targets = shift_tokens.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_row_loss = per_token_loss.view(tokens.size(0), -1)

    # restrict to completion region (mask shifted)
    shift_mask = mask[..., 1:].contiguous()
    masked_loss = per_row_loss * shift_mask

    loss_sum = masked_loss.sum(dim=1)
    loss_mean = loss_sum / shift_mask.sum(dim=1)
    return int(loss_mean.argmin().item())


# -----------------------------------------------------------------------------


# DDP initialization and device selection
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

IS_DDP = int(os.environ.get("RANK", -1)) != -1
if IS_DDP:
    assert torch.cuda.is_available(), "DDP path requires CUDA for this script"
    init_process_group(backend="nccl")
    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{LOCAL_RANK}"
    torch.cuda.set_device(DEVICE)
    MASTER_PROCESS = RANK == 0
else:
    RANK = 0
    LOCAL_RANK = 0
    WORLD_SIZE = 1
    MASTER_PROCESS = True
    # autodetect device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"using device: {DEVICE}")

DEVICE_TYPE = "cuda" if str(DEVICE).startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

tokenizer = tiktoken.get_encoding("gpt2")

# Global training settings
TOTAL_BATCH_TOKENS = 524_288  # ~0.5M tokens
MICRO_BATCH = 64
SEQ_LEN = 1024
assert (
    TOTAL_BATCH_TOKENS % (MICRO_BATCH * SEQ_LEN * WORLD_SIZE) == 0
), "TOTAL_BATCH_TOKENS must be divisible by MICRO_BATCH*SEQ_LEN*WORLD_SIZE"
GRAD_ACCUM_STEPS = TOTAL_BATCH_TOKENS // (MICRO_BATCH * SEQ_LEN * WORLD_SIZE)
if MASTER_PROCESS:
    print(f"total desired batch size (tokens): {TOTAL_BATCH_TOKENS}")
    print(f"=> gradient accumulation steps: {GRAD_ACCUM_STEPS}")

train_loader = DataLoaderLite(
    batch_size=MICRO_BATCH, seq_len=SEQ_LEN, process_rank=RANK, world_size=WORLD_SIZE, split="train"
)
val_loader = DataLoaderLite(
    batch_size=MICRO_BATCH, seq_len=SEQ_LEN, process_rank=RANK, world_size=WORLD_SIZE, split="val"
)

torch.set_float32_matmul_precision("high")

# Model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2")
model.to(DEVICE)

USE_COMPILE = False  # torch.compile can interfere with eval/generation here
if USE_COMPILE:
    model = torch.compile(model)

if IS_DDP:
    model = DDP(model, device_ids=[LOCAL_RANK])

raw_model = model.module if IS_DDP else model

# LR schedule
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
MAX_STEPS = 19_073  # ~1 epoch for 10B tokens at 0.5M tokens/step


def learning_rate_for(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# Optimizer
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=DEVICE_TYPE, is_master_process=MASTER_PROCESS
)

# Logging
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
with open(LOG_FILE, "w"):
    pass  # truncate


for step in range(MAX_STEPS):
    t_start = time.time()
    is_last_step = step == MAX_STEPS - 1

    # ---- Validation
    if step % 250 == 0 or is_last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                xb, yb = val_loader.next_batch()
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
                val_loss_accum += (loss / val_loss_steps).detach()
        if IS_DDP:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if MASTER_PROCESS:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(LOG_FILE, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or is_last_step):
                ckpt_path = os.path.join(LOG_DIR, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, ckpt_path)

    # ---- HellaSwag eval
    if (step % 250 == 0 or is_last_step) and (not USE_COMPILE):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iter_examples("val")):
            if i % WORLD_SIZE != RANK:
                continue
            _, tokens, mask, label = render_multiple_choice_tensors(example)
            tokens = tokens.to(DEVICE)
            mask = mask.to(DEVICE)
            with torch.no_grad():
                with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
            pred_idx = most_likely_completion_index(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_idx == label)

        if IS_DDP:
            num_total = torch.tensor(num_total, dtype=torch.long, device=DEVICE)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=DEVICE)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = int(num_total.item())
            num_correct_norm = int(num_correct_norm.item())

        acc_norm = num_correct_norm / max(1, num_total)
        if MASTER_PROCESS:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(LOG_FILE, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # ---- Text generation (skip step 0)
    if ((step > 0 and step % 250 == 0) or is_last_step) and (not USE_COMPILE):
        model.eval()
        num_return_sequences = 4
        max_len = 32
        prompt_ids = tokenizer.encode("Hello, I'm a language model,")
        xb = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = xb.to(DEVICE)

        gen_rng = torch.Generator(device=DEVICE)
        gen_rng.manual_seed(42 + RANK)
        while xgen.size(1) < max_len:
            with torch.no_grad():
                with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16):
                    logits, _ = model(xgen)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                sampled = torch.multinomial(topk_probs, 1, generator=gen_rng)
                next_tok = torch.gather(topk_indices, -1, sampled)
                xgen = torch.cat((xgen, next_tok), dim=1)

        for i in range(num_return_sequences):
            out_ids = xgen[i, :max_len].tolist()
            print(f"rank {RANK} sample {i}: {tokenizer.decode(out_ids)}")

    # ---- Training step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(GRAD_ACCUM_STEPS):
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if IS_DDP:
            model.require_backward_grad_sync = (micro_step == GRAD_ACCUM_STEPS - 1)
        with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        loss = loss / GRAD_ACCUM_STEPS
        loss_accum += loss.detach()
        loss.backward()

    if IS_DDP:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = learning_rate_for(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if DEVICE_TYPE == "cuda":
        torch.cuda.synchronize()

    dt = time.time() - t_start
    tokens_this_step = train_loader.batch_size * train_loader.seq_len * GRAD_ACCUM_STEPS * WORLD_SIZE
    toks_per_sec = tokens_this_step / max(1e-8, dt)

    if MASTER_PROCESS:
        print(
            f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
            f"norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {toks_per_sec:.2f}"
        )
        with open(LOG_FILE, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if IS_DDP:
    destroy_process_group()
