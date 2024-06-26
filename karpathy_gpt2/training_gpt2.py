import os
from os import getenv
from sys import exit  # for debugging
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from tinygrad import Tensor, device, dtypes, TinyJit, Device
from tinygrad.helpers import fetch
from tinygrad.nn import Embedding, LayerNorm, Linear
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import (
    get_parameters,
    get_state_dict,
    load_state_dict,
    torch_load,
)
from tqdm import tqdm, trange

# TODO: TF32?
# TODO: DDP
# TODO: Mixed-precision - + expand attention if necessary to do mixed precision on dot
# TODO: Clip grad norm to 1 while accounting for multiple devices
# TODO: FusedAdamW?
# TODO: FineWebEdu

# for now, mixed precision toggle
MP = True

# ---------------- UTILS ----------------


# taken from https://github.com/tinygrad/tinygrad/blob/97b05f567e8e42a2475f8a063fb080b200f6f033/extra/models/mask_rcnn.py
def topk(input_, k, device, dim=-1, largest=True, sorted=False):
    k = min(k, input_.shape[dim] - 1)
    input_ = input_.numpy()
    if largest:
        input_ *= -1
    ind = np.argpartition(input_, k, axis=dim)
    if largest:
        input_ *= -1
    ind = np.take(ind, np.arange(k), axis=dim)  # k non-sorted indices
    input_ = np.take_along_axis(input_, ind, axis=dim)  # k non-sorted values
    if not sorted:
        return Tensor(input_, device=device), Tensor(ind, device=device)
    if largest:
        input_ *= -1
    ind_part = np.argsort(input_, axis=dim)
    ind = np.take_along_axis(ind, ind_part, axis=dim)
    if largest:
        input_ *= -1
    val = np.take_along_axis(input_, ind_part, axis=dim)
    return Tensor(val, device=device), Tensor(ind, device=device)


# ---------------- GPT CONFIG ----------------


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5


@dataclass
class GPT2Small(GPT2Config):
    pass


@dataclass
class GPT2Medium(GPT2Config):
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024


@dataclass
class GPT2Large(GPT2Config):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280


@dataclass
class GPT2XL(GPT2Config):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600


MODEL_CONFIGS = {
    "gpt2": GPT2Small,
    "gpt2-medium": GPT2Medium,
    "gpt2-large": GPT2Large,
    "gpt2-xl": GPT2XL,
}

# --------------- MODEL ----------------


class MLP:
    def __init__(self, config: GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.RESIDUAL_SCALING = 1

    @property
    def parameters(self):
        return [self.c_fc, self.c_proj]

    def __call__(self, x):
        x = self.c_fc(x)
        x = x.gelu()
        x = self.c_proj(x)
        return x


class Attention:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.c_attn = Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALING = 1

    @property
    def parameters(self):
        return [self.c_attn, self.c_proj]

    def __call__(self, x):
        B, T, C = x.shape
        dtype = x.dtype

        q, k, v = self.c_attn(x).split(C, dim=-1)  # (B,T,3C) -> (B,T,C) x 3
        split_heads = lambda x: x.view(
            B, T, self.config.n_head, self.config.n_embd // self.config.n_head
        ).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class TransformerBlock:
    def __init__(self, config: GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)

    @property
    def parameters(self):
        return [self.ln_1, self.ln_2, *self.attn.parameters, *self.mlp.parameters]

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2:
    def __init__(self, config: GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights - HUGE SAVINGS
        self.lm_head.weight = self.wte.weight

        # init weights
        for param in self.parameters:
            self.init_weights(param)

    @property
    def parameters(self):
        parameters = [self.wte, self.wpe, self.ln_f, self.lm_head]
        for block in self.h:
            parameters.extend(block.parameters)
        return parameters

    def init_weights(self, param):
        if isinstance(param, Linear):
            std = 0.02
            # apply residual scaling
            if hasattr(param, "RESIDUAL_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            param.weight = Tensor.normal(
                param.weight.shape,
                mean=0,
                std=std,
                dtype=(dtypes.bfloat16 if MP else dtypes.float32),
            )
            if param.bias is not None:
                param.bias = Tensor.zeros_like(
                    param.bias, dtype=(dtypes.bfloat16 if MP else dtypes.float32)
                )
        elif isinstance(param, Embedding):
            param.weight = Tensor.normal(param.weight.shape, mean=0, std=0.02)

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        assert (
            T <= self.config.block_size
        ), f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long, device=GPUS)  # (T,)
        pos_emb = self.wpe(pos)  # (T,) -> (T,C)
        tok_emb = self.wte(idx)  # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        dtype = x.dtype
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,C) -> (B,T,V)

        if targets is not None:
            loss = logits.flatten(0, 1).sparse_categorical_crossentropy(
                targets.flatten()
            )
            return logits, loss.realize()

        return logits, None

    def generate(self, prompt: str, tokenizer, max_length, num_return_sequences):
        tokens = tokenizer.encode(prompt)
        x = (
            Tensor(tokens, dtype=dtypes.long, device=GPUS)
            .unsqueeze(0)
            .repeat(num_return_sequences, 1)
        )
        Tensor.no_grad = True
        Tensor.training = False
        while x.shape[1] < max_length:
            logits, _ = self(x)
            logits = logits[:, -1, :]
            probs = logits.softmax(-1)
            topk_probs, topk_indices = topk(probs, 50, GPUS, dim=-1)
            ix = topk_probs.multinomial(1)
            xcol = topk_indices.gather(-1, ix)
            x = x.cat(xcol, dim=1)

        for i in range(num_return_sequences):
            tokens = x[i, :max_length].numpy().tolist()
            decoded = tokenizer.decode(tokens)
            print(">", decoded)

    @staticmethod
    def build(MODEL_NAME):
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        weights = torch_load(
            fetch(f"https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin")
        )

        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights["lm_head.weight"] = weights["wte.weight"]
        load_state_dict(model, weights)

        return model


# ---------------- MULTI GPU ----------------

GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
NUM_GPUS = len(GPUS)

# ---------------- DATA LOADER ----------------


class DataLoaderLite:
    def __init__(self, B, T, file_path):
        self.B = B
        self.T = T

        self.batch = lambda x: x.view(B, T)

        with open(file_path, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")

        tokens = enc.encode(text)
        self.tokens = Tensor(tokens, dtype=dtypes.long)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = self.batch(buf[:-1]).shard_(GPUS, axis=0)
        y = self.batch(buf[1:]).shard_(GPUS, axis=0)
        self.current_position += B * T * NUM_GPUS

        if self.current_position + (B * T * NUM_GPUS + 1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x, y


# ---------------- OPTIMIZER ----------------


def create_optimizers(model, **optim_args):
    # TODO: do I need to include requires_grad for the count to be correct?
    # Think about this when adding bias

    params_nodecay = [param for param in get_parameters(model) if len(param.shape) < 2]
    params_decay = [param for param in get_parameters(model) if len(param.shape) >= 2]
    num_params_decay = sum(param.numel() for param in params_decay)
    num_params_nodecay = sum(param.numel() for param in params_nodecay)
    print(
        f"num decay params: {num_params_decay} num nodecay params: {num_params_nodecay}"
    )
    opt_decay = AdamW(params_decay, **optim_args, weight_decay=0.1)
    opt_nodecay = AdamW(params_nodecay, **optim_args, weight_decay=0)

    optim_group = OptimizerGroup(opt_decay, opt_nodecay)

    return optim_group


# ---------------- INITIALIZATION ----------------

B = 2
T = 1024
total_batch_size = 2**12  # ~.5M, measured in tokens
num_epochs = 100
optim_args = {
    "lr": 6e-4,
    "b1": 0.9,
    "b2": 0.95,
    "eps": 1e-8,
}

# grad accum
assert total_batch_size % (B * T) == 0, "total batch size must be divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"grad_accum_steps = {grad_accum_steps}")


tokenizer = tiktoken.get_encoding("gpt2")
dl = DataLoaderLite(B, T, "datasets/shake.txt")
model = GPT2(GPT2Small(vocab_size=50304))
for k, x in get_state_dict(model).items():
    x.to_(GPUS)
optim_group = create_optimizers(model, **optim_args)


# ---------------- LR ----------------


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = (
    19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return Tensor(min_lr + coeff * (max_lr - min_lr), requires_grad=False)


# ---------------- TRAINING ----------------


@TinyJit
def train_step():
    x, y = dl.next_batch()
    optim_group.zero_grad()
    logits, loss = model(x, y)
    loss = (
        loss / grad_accum_steps
    )  # TODO: is it bad style to reference grad_accum_steps here?
    loss.backward()
    return loss.numpy()


avg_dt = 0
avg_tokens_per_sec = 0
losses = []
for step in range(num_epochs):
    last_step = step == max_steps - 1
    if (step > 0 and step % 250 == 0) or last_step:
        model.generate(
            "Hello, I'm a language model,",
            tokenizer,
            max_length=30,
            num_return_sequences=2,
        )
    Tensor.training = True
    Tensor.no_grad = False
    t0 = time.perf_counter()
    full_batch_loss = 0
    for micro_step in range(grad_accum_steps):
        loss = train_step()
        full_batch_loss += loss
    # clip grads here
    lr = get_lr(step)
    # TODO: be careful about device with lr here?
    for optim in optim_group.optimizers:
        optim.lr = lr
    optim_group.step()
    loss = train_step(grad_accum_steps)
    losses.append(loss)
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (dl.B * dl.T * grad_accum_steps) / (t1 - t0)
    avg_dt += dt
    avg_tokens_per_sec += tokens_per_sec
    print(
        f"step: {step} | lr: {lr:.5f} | loss {full_batch_loss} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}"
    )
print(
    f"avg dt: {avg_dt/num_epochs:.2f}ms avg tokens/sec: {avg_tokens_per_sec/num_epochs:.2f}"
)
