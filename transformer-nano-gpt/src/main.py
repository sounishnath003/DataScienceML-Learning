"""
# _* coding: utf8 *_

filename: main.py

@credits: Andrej Karpathy - NanoGPT Hack into Tranformers GPT
@author: sounishnath
createdAt: 2023-03-29 21:32:32
"""

import logging

import torch.nn as nn

import torch
from src.config import Configuration


class Head(nn.Module):
    """one head of self attention node"""

    def __init__(self, n_embedding_size, n_head_size, *args, **kwargs) -> None:
        super(Head, self).__init__()
        self.key = nn.Linear(n_embedding_size, n_head_size, bias=False)
        self.query = nn.Linear(n_embedding_size, n_head_size, bias=False)
        self.value = nn.Linear(n_embedding_size, n_head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(Configuration.block_size, Configuration.block_size)),
        )
        self.dropout = nn.Dropout(Configuration.dropout)

    def forward(self, X):
        B, T, C = X.shape
        K = self.key(X)
        Q = self.key(X)
        # attention process is completely depends upon the ML engineering how they want to tweak the process
        # In genral here's a simple standard code of weights affinities by Andrej Karpathy (Tesla.ML)
        # compute attention scores
        Wei = torch.matmul(Q, K.transpose(-2, -1) * C**-0.50)
        Wei = Wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        Wei = nn.functional.softmax(Wei, dim=-1)  # (B, T, T)
        Wei = self.dropout(Wei)
        V = self.value(X)
        out = Wei @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, *args, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embedding_size=Configuration.embedding_size,
                    n_head_size=head_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(
            Configuration.embedding_size, Configuration.embedding_size
        )
        self.dropout = nn.Dropout(Configuration.dropout)

    def forward(self, X):
        logits = torch.cat([head(X) for head in self.heads], dim=-1)
        logits = self.dropout(self.proj(logits))
        return logits


class FeedForwardNetwork(nn.Module):
    """simple linear layer followed by non-linearity"""

    def __init__(self, *args, **kwargs) -> None:
        super(FeedForwardNetwork, self).__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(Configuration.embedding_size, 4 * Configuration.embedding_size),
            nn.ReLU(),
            nn.Linear(4 * Configuration.embedding_size, Configuration.embedding_size),
            nn.Dropout(Configuration.dropout),
        )

    def forward(self, X):
        return self.network(X)


class Block(nn.Module):
    """entire transformer architecture core but without cross-attention head"""

    def __init__(self, n_embedding_size, n_heads, *args, **kwargs) -> None:
        super(Block, self).__init__(*args, **kwargs)
        self.embedding_size = n_embedding_size
        self.n_heads = n_heads
        head_size = n_embedding_size // n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size)
        self.ffd = FeedForwardNetwork()
        self.ln1 = nn.LayerNorm(Configuration.embedding_size)
        self.ln2 = nn.LayerNorm(Configuration.embedding_size)

    def forward(self, X):
        X = X + self.sa(self.ln1(X))
        logits = X + self.ffd(self.ln2(X))
        return logits


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, Configuration.embedding_size
        )
        self.position_embedding_table = nn.Embedding(
            Configuration.block_size, Configuration.embedding_size
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    Configuration.embedding_size,
                    n_heads=Configuration.n_attention_heads,
                )
                for _ in range(Configuration.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(Configuration.embedding_size)  # final layer norm
        self.lm_head = nn.Linear(Configuration.embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=Configuration.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -Configuration.block_size :]
            # get the predictions
            logits, loss = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(1331)
    logging.basicConfig(level=logging.INFO)

    with open(Configuration.dataset_path, encoding="utf-8") as file:
        raw_text = file.read()

    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)
    logging.info("".join(chars))
    logging.info(vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda s: "".join([itos[c] for c in s])

    s = "i am a good boy!"
    logging.info(f"original_text={s}")
    logging.info(f"encode_text={encode(s)}")

    # train/valid splitting
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    n = int(0.80 * len(data))  # 80:20 ratio train:valid
    train_data = data[:n]
    valid_data = data[n:]

    def get_batch(split="train"):
        global train_data, valid_data
        data = train_data if split == "train" else valid_data
        ix = torch.randint(
            len(data) - Configuration.block_size, (Configuration.batch_size,)
        )

        x = torch.stack([data[i : i + Configuration.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + Configuration.block_size + 1] for i in ix])
        x, y = x.to(Configuration.device), y.to(Configuration.device)
        return x, y

    x, y = get_batch(split="train")
    logging.info(x.shape)
    logging.info(y.shape)

    model = BigramLanguageModel()
    logging.info(model)
    m = model.to(Configuration.device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "valid"]:
            losses = torch.zeros(Configuration.eval_iteration)
            for k in range(Configuration.eval_iteration):
                X, y = get_batch(split)
                logits, loss = model(X, y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        model.train()
        return out

    # print the number of parameters in the model
    logging.info(f"{sum(p.numel() for p in m.parameters()) / 1e6} Million parameters.")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Configuration.learning_rate)

    for iter in range(Configuration.epochs):
        # every once in a while evaluate the loss on train and val sets
        if iter % Configuration.eval_intervals == 0 or iter == Configuration.epochs - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=Configuration.device)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
