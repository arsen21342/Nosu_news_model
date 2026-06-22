import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from tokenizers import Tokenizer

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class OssetianDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.texts.append(data["text"])
        
        self.encoded_texts = []
        for text in self.texts:
            encoded = tokenizer.encode(text).ids
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            self.encoded_texts.append(encoded)
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        input_ids = self.encoded_texts[idx]
        target_ids = input_ids[1:] + [0]
        
        pad_token_id = 1 
        while len(input_ids) < self.max_length:
            input_ids.append(pad_token_id)
            target_ids.append(pad_token_id)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def create_dataloader(jsonl_file, tokenizer, batch_size, max_length, shuffle=True):
    dataset = OssetianDataset(jsonl_file, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits = logits.view(-1, logits.size(-1))
    targets = target_batch.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=1)
    return loss

def train_model(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

def generate_text(model, tokenizer, prompt, max_new_tokens, context_size, device):
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
    generated_ids = input_ids.squeeze(0).tolist()
    return tokenizer.decode(generated_ids)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file("ossetian_bpe_tokenizer/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    config = {
        "vocab_size": vocab_size,
        "context_length": 128,      
        "emb_dim": 128,            
        "n_heads": 4,             
        "n_layers": 4,            
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    print("Model config:", config)
    
    model = GPTModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("Loading dataset...")
    batch_size = 8
    train_loader = create_dataloader(
        "ossetian_corpus.jsonl",
        tokenizer,
        batch_size=batch_size,
        max_length=config["context_length"],
        shuffle=True
    )
    print(f"Number of batches: {len(train_loader)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    
    print("Starting training...")
    num_epochs = 50
    train_model(model, train_loader, optimizer, device, num_epochs)
    
    print("\n--- Testing generation ---")
    prompt = "ЦИПУ-йы"
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        context_size=config["context_length"],
        device=device
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    
    torch.save(model.state_dict(), "ossetian_gpt_model.pth")
    print("Model saved as ossetian_gpt_model.pth")

if __name__ == "__main__":
    main()