import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparams
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3_000
EVAL_INTERVAL = 300  # model will be evaluated after every EVAL_INTERVAL iterations
EVAL_ITERS = 200  # model's loss will be calcuated this many times during evaluation
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # to use GPU if possible


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


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_emb_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
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
