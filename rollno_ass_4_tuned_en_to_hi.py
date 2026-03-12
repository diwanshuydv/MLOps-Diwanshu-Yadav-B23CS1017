"""
Assignment 4: Optimizing Transformer Translation with Ray Tune & Optuna
=======================================================================
This script refactors the en_to_hi.ipynb notebook to use Ray Tune with
OptunaSearch for hyperparameter optimization of a from-scratch Transformer
model that translates English to Hindi.

Baseline (100 epochs): Final Loss ~0.0974, BLEU ~52.47%
Goal: Match or exceed baseline BLEU in ≤50 epochs via hyperparameter tuning.

Hyperparameters tuned (5):
  1. Learning Rate (lr)         — loguniform(1e-5, 1e-3)
  2. Batch Size (batch_size)    — choice([32, 60, 64, 128])
  3. Num Attention Heads        — choice([4, 8])
  4. FeedForward Dim (d_ff)     — choice([1024, 2048])
  5. Dropout Rate (dropout)     — uniform(0.1, 0.4)
"""

import os
import math
import time
import pickle
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "English-Hindi.tsv")
MAX_LEN = 50
D_MODEL = 512  # Fixed — must be divisible by num_heads choices (4, 8)
NUM_LAYERS = 6  # Fixed


def load_data():
    """Load and clean the English-Hindi TSV dataset."""
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# 2. Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def tokenize(self, sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])


def encode_sentence(sentence, vocab, max_len=50):
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len - 2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))


# ---------------------------------------------------------------------------
# 3. Transformer model (identical to notebook)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.query_linear(q)
        K = self.key_linear(k)
        V = self.value_linear(v)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_layers=6, num_heads=8, d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size))).bool().to(next(self.parameters()).device)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        out = self.fc_out(dec_out)
        return out


# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output


# ---------------------------------------------------------------------------
# 5. Translation & BLEU evaluation helpers
# ---------------------------------------------------------------------------

def translate_sentence(model, sentence, en_vocab, hi_vocab, device, max_len=50):
    model.eval()
    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]
    tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tokens = [hi_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_pad_idx, tgt_pad_idx)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break
    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]]
    return " ".join(translated)


def evaluate_bleu(model, en_vocab, hi_vocab, device):
    """Evaluate BLEU on a small validation set (same as notebook)."""
    smoothie = SmoothingFunction().method4

    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।"),
        ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
        ("Let me tell Tom.", "मुझे टॉम को बताने दीजिए।"),
    ]

    references = []
    hypotheses = []
    for en_sentence, hi_sentence in val_dataset:
        pred = translate_sentence(model, en_sentence, en_vocab, hi_vocab, device)
        pred_tokens = pred.split()
        ref_tokens = hi_sentence.split()
        references.append([ref_tokens])
        hypotheses.append(pred_tokens)

    score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return score


# ---------------------------------------------------------------------------
# 6. Build vocab (shared across trials)
# ---------------------------------------------------------------------------

def build_vocabs():
    """Build vocabularies from the dataset (done once before tuning)."""
    df = load_data()
    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())
    return df, en_vocab, hi_vocab


# ---------------------------------------------------------------------------
# 7. train_tune: the Ray Tune trainable function
# ---------------------------------------------------------------------------

def train_tune(config):
    """
    Training function for a single Ray Tune trial.
    Reads hyperparameters from `config` dict.
    """
    # --- Hyperparameters from config ---
    lr = config["lr"]
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]
    num_epochs = config["num_epochs"]

    # --- Load pre-built vocabs & data (passed via config) ---
    df = load_data()

    en_vocab = config["en_vocab"]
    hi_vocab = config["hi_vocab"]

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset & DataLoader ---
    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # --- Model ---
    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=num_heads,
        d_ff=d_ff,
        max_len=MAX_LEN,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in train_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Compute BLEU every 10 epochs (expensive) or on the last epoch
        bleu = -1.0
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            bleu = evaluate_bleu(model, en_vocab, hi_vocab, device)

        # Report to Ray Tune (ASHA scheduler uses this to decide early stopping)
        ray.train.report({"loss": avg_loss, "bleu": bleu})


# ---------------------------------------------------------------------------
# 8. Main: Configure & run the hyperparameter sweep
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Assignment 4: Optimizing Transformer Translation with Ray Tune & Optuna")
    print("=" * 70)

    # Build vocabs once (shared across all trials)
    print("\n[1/4] Loading data and building vocabularies...")
    df, en_vocab, hi_vocab = build_vocabs()
    print(f"  English vocab size: {len(en_vocab)}")
    print(f"  Hindi vocab size:   {len(hi_vocab)}")

    # --- Search space ---
    MAX_EPOCHS_PER_TRIAL = 50

    search_space = {
        # Tunable hyperparameters
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 60, 64, 128]),
        "num_heads": tune.choice([4, 8]),
        "d_ff": tune.choice([1024, 2048]),
        "dropout": tune.uniform(0.1, 0.4),
        # Fixed
        "num_epochs": MAX_EPOCHS_PER_TRIAL,
        "en_vocab": en_vocab,
        "hi_vocab": hi_vocab,
    }

    # --- Optuna search algorithm ---
    optuna_search = OptunaSearch(metric="loss", mode="min")

    # --- ASHA scheduler for early stopping ---
    asha_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_EPOCHS_PER_TRIAL,
        grace_period=5,
        reduction_factor=3,
    )

    # --- Run the sweep ---
    print(f"\n[2/4] Starting Ray Tune sweep (up to {MAX_EPOCHS_PER_TRIAL} epochs/trial)...")

    tuner = tune.Tuner(
        train_tune,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            num_samples=20,
            max_concurrent_trials=1,  # Adjust based on GPU memory
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    # --- Get best result ---
    best_result = results.get_best_result(metric="loss", mode="min")
    best_config = best_result.config

    print("\n[3/4] ✅ Best configuration found:")
    print(f"  lr:         {best_config['lr']:.6f}")
    print(f"  batch_size: {best_config['batch_size']}")
    print(f"  num_heads:  {best_config['num_heads']}")
    print(f"  d_ff:       {best_config['d_ff']}")
    print(f"  dropout:    {best_config['dropout']:.4f}")
    print(f"  Best loss:  {best_result.metrics['loss']:.4f}")
    if best_result.metrics.get("bleu", -1) > 0:
        print(f"  BLEU:       {best_result.metrics['bleu'] * 100:.2f}%")

    # --- Retrain best model and save ---
    print("\n[4/4] Retraining best model for final evaluation & saving...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=best_config["num_heads"],
        d_ff=best_config["d_ff"],
        max_len=MAX_LEN,
        dropout=best_config["dropout"],
    ).to(device)

    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(best_model.parameters(), lr=best_config["lr"])

    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(
        dataset, batch_size=best_config["batch_size"], shuffle=True, collate_fn=collate_fn
    )

    start_time = time.time()
    final_loss = 0

    for epoch in range(MAX_EPOCHS_PER_TRIAL):
        best_model.train()
        epoch_loss = 0
        for src, tgt_input, tgt_output in train_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = best_model(src, tgt_input, src_pad_idx, tgt_pad_idx)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        final_loss = avg_loss
        print(f"  Epoch [{epoch+1}/{MAX_EPOCHS_PER_TRIAL}] — Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time

    # Final BLEU
    final_bleu = evaluate_bleu(best_model, en_vocab, hi_vocab, device)

    print(f"\n{'=' * 50}")
    print(f"  Final Loss:  {final_loss:.4f}")
    print(f"  Final BLEU:  {final_bleu * 100:.2f}%")
    print(f"  Time:        {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 50}")

    # Save model weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "rollno_ass_4_best_model.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"\n✅ Best model saved to: {model_path}")

    # Save vocabs
    with open(os.path.join(script_dir, "en_vocab.pkl"), "wb") as f:
        pickle.dump(en_vocab, f)
    with open(os.path.join(script_dir, "hi_vocab.pkl"), "wb") as f:
        pickle.dump(hi_vocab, f)
    print("✅ Vocabularies saved.")

    # Print some example translations
    print("\n📝 Example translations:")
    test_sentences = [
        "I love you.",
        "What is your name?",
        "How are you?",
        "The weather is nice today.",
        "She is a good teacher.",
    ]
    for sentence in test_sentences:
        translation = translate_sentence(best_model, sentence, en_vocab, hi_vocab, device)
        print(f"  EN: {sentence}")
        print(f"  HI: {translation}")
        print()


if __name__ == "__main__":
    main()
