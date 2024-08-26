import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparams
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5_000
EVAL_INTERVAL = 500  # model will be evaluated after every EVAL_INTERVAL iterations
EVAL_ITERS = 200  # model's loss will be calcuated this many times during evaluation
LEARNING_RATE = 1e-3  # this was 1e-2 previously, Self-Attention can't tolerate large learning rates
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # to use GPU if possible

N_EMBD = 32  # embedding dimension


torch.manual_seed(1337)


def read_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def make_chars_and_vocab_size(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    return chars, vocab_size

def make_mappings(chars):
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for ch, i in stoi.items()}
    return stoi, itos

def make_enc_dec(stoi, itos):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

def train_test(ratio, data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def train(model, optimizer, get_batch):
    for iter in range(MAX_ITERS):
        # evaluate after every EVAL_INTERVAL iterations
        if iter % EVAL_INTERVAL == 0:
            # before, we were printing loss at every iteration
            # but that is noisy (some batches can get lucky and have low loss)
            # so we print the mean loss instead, which is less noisy
            losses = estimate_loss(model, get_batch)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def generate(model, max_new_tokens):
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # need to keep input on same device as model (CPU or GPU)
        return decode(model.generate(context, max_new_tokens)[0].tolist())

# this is a Context Manager
# disables gradient calculation for this function (as we don't update model parameters)
# makes it more efficient for memory (we don't store intermediate vars used in loss.backward())
@torch.no_grad()
def estimate_loss(model, get_batch):
    out = {}
    # put model in eval mode (no training)
    # this won't do anything for our BigramLM, as the model just uses a table
    # (no dropout or other such layers which need to behave differently in evaluation compared to during training)
    model.eval()
    for split in ['train', 'val']:
        # store total of EVAL_ITERS number of losses
        losses = torch.zeros(EVAL_ITERS)
        # calculate and store loss for each iteration
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # get mean loss for current split
        out[split] = losses.mean()
    model.train()  # put model back in training mode
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        # register_buffer is used to create variables that are not model parameters
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.register_buffer('head_size', torch.ones(1) * head_size)

    def forward(self, x):
        B, T, C = x.shape

        # (B, T, C)
        k = self.key(x)
        q = self.query(x)

        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # this makes it a Decoder block
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        # concat over the channel dim (the C in (B, T, C))
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, N_EMBD)
        # ths is BLOCK_SIZE by N_EMBD
        # why BLOCK_SIZE, why not vocab_size here? we want positional embeddings for each position in the input vector -> length of vector != size of vocab
        self.pos_emb_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        # because we have 4 communication channels now (4 heads), head_size would typically be smaller
        # 4 heads and each give 8-dim vector, so result after concatenating is 4*8 = 32-dim vector (in channel dim)
        # this is like group convolutions: instead of 1 large conv, we do multiple smaller convs
        self.sa_heads = MultiHeadAttention(4, N_EMBD // 4)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # now, table doesn't give us logits directly
        # it gives us embeddings of the tokens (this is how we interpret it now, it's still just some numbers)
        token_emb = self.token_emb_table(idx)  # (B, T, C)
        # torch.arange() gives numbers 0 to T-1
        # all these numbers will be embedded through the table
        pos_emb = self.pos_emb_table(torch.arange(T, device=DEVICE))  # (T, C)
        # now x contains pos info too
        # Note: this pos info isn't useful for BigramLM
        # Bigram just uses prev token, so it doesn't matter whether you're at position 5 or some other (it is "translation invariant")
        x = token_emb + pos_emb  # (B, T, C), pos_emb will be broadcasted across batch (i.e. (T, C) -> (B, T, C))
        
        # apply 1 head of Self-Attention
        # (B, T, C)
        x = self.sa_heads(x)
        
        logits = self.lm_head(x)  # (B, T, vocab_size C)
        
        if targets is None:
            loss = None
        else:
            B, T, vocab_size_C = logits.shape
            logits = logits.view(B*T, vocab_size_C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_in = idx[:, -BLOCK_SIZE:]  # only feed last BLOCK_SIZE tokens, to avoid out-of-bounds for our positional embedding table
            logits, loss = self(idx_in)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


if __name__ == "__main__":
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        
        # need to keep data and model on same device (CPU or GPU)
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y
    
    
    print(f"Using device={DEVICE}")

    data = read_data()
    chars, vocab_size = make_chars_and_vocab_size(data)
    
    stoi, itos = make_mappings(chars)
    encode, decode = make_enc_dec(stoi, itos)

    data = torch.tensor(encode(data), dtype=torch.long)
    train_data, val_data = train_test(0.9, data)

    model = BigramLM(vocab_size)
    # need to keep data and model on same device (CPU or GPU)
    m = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer, get_batch)

    print(generate(model, 500))
