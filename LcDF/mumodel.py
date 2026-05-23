import torch
import torch.nn as nn
import torch.nn.functional as F

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + position_emb + token_type_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out(context)

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size=2048, dropout=0.1):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class BertEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class BertForSentiment(nn.Module):
    def __init__(self, num_classes, vocab_size=21128, hidden_size=768, num_heads=12, num_layers=12, max_seq_len=128):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_seq_len)
        self.encoder = BertEncoder(num_layers, hidden_size, num_heads, hidden_size * 4)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
        x = self.encoder(x, attention_mask)
        cls_token = x[:, 0, :]
        return self.head(cls_token)

def model_base_12(num_classes, max_seq_len=128, vocab_size=21128):
    return BertForSentiment(
        num_classes=num_classes,
        vocab_size=vocab_size,
        num_layers=12,
        max_seq_len=max_seq_len
    )

def model_small_6(num_classes, max_seq_len=128, vocab_size=21128):
    return BertForSentiment(
        num_classes=num_classes,
        vocab_size=vocab_size,
        num_layers=6,
        max_seq_len=max_seq_len
    )

def model_small_4(num_classes, max_seq_len=128, vocab_size=21128):
    return BertForSentiment(
        num_classes=num_classes,
        vocab_size=vocab_size,
        num_layers=4,
        max_seq_len=max_seq_len
    )