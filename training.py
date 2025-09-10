import torch
import torch.nn as nn
from torch.nn import functional as F

# Запуск на CPU не лучшая идея
# параметры
batch_size = 64
block_size = 256 # context of self attention
max_iters = 5000 #epochs
eval_interval = 500 # logs every n
learning_rate = 1e-3 #step in opt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 256 # size of emb
dropout = 0.2 # droupout param
# ------------

torch.manual_seed(1337)

with open('gpt_dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to ntegers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_emb, n_emb*4),
            nn.ReLU(),
            nn.Linear(n_emb*4, n_emb),
            nn.Dropout(0.2)

        )
    def forward(self,x):
        return self.layers(x)

class  Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
    def forward(self,x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        tril = torch.tril(torch.ones(block_size,block_size, device=device))
        wei = wei.masked_fill(tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,head_size,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size,num_heads*head_size)
        self.droupout = nn.Dropout(0.2)
    def forward(self,x):
        out =  torch.cat([h(x) for h in self.heads] , dim = -1)
        return self.droupout(self.proj(out))

class Block(nn.Module):
 def __init__(self,n_emb,n_head):
     super().__init__()
     self.sa = MultiHeadAttention(n_emb // n_head,n_head)
     self.ffwd = FeedForward(n_emb)
     self.ln1 = nn.LayerNorm(n_emb)
     self.ln2 = nn.LayerNorm(n_emb)

 def forward(self,x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        # self.up_matrix = torch.tril(torch.ones(n_emb, n_emb, device=device))
        # self.up_matrix = F.softmax(self.up_matrix.masked_fill(self.up_matrix == 0, -torch.inf), dim = -1)
        self.lm_head = nn.Linear(n_emb,vocab_size)
        self.blocks = nn.Sequential(
            Block(n_emb,n_head=4),
            Block(n_emb,n_head=4),
            Block(n_emb,n_head=4),
            Block(n_emb,n_head=4),
            Block(n_emb,n_head=4),

            nn.LayerNorm(n_emb),
        )

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        poss_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = poss_emb+token_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
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
            # get the predictions
            idx_new = idx[:,-block_size:]
            logits, loss = self(idx_new)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
torch.save(m.state_dict(), "gpt-weights.tar")
