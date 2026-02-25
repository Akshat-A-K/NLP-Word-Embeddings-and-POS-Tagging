import nltk
import torch
import numpy as np
from nltk.corpus import brown
from collections import Counter
import os

# -----------------------------
# 1. Load Brown POS data
# -----------------------------
nltk.download("brown")

tagged_sents = brown.tagged_sents(tagset="universal")

# Build vocab ONLY from training split (same as POS pipeline)
num_sentences = len(tagged_sents)
train_sents = tagged_sents[:int(0.8 * num_sentences)]

all_words = []
for sent in train_sents:
    for word, tag in sent:
        all_words.append(word.lower())

word_counts = Counter(all_words)
vocab = [w for w, c in word_counts.items() if c >= 2]

# Add PAD token
vocab = ["<PAD>"] + sorted(vocab)

word_to_index = {w: i for i, w in enumerate(vocab)}

print(f"Brown vocab size (incl PAD): {len(word_to_index)}")

# -----------------------------
# 2. Load raw GloVe
# -----------------------------
glove_path = "glove.6B.300d.txt"
embed_dim = 300

print("Loading raw GloVe vectors...")
glove_raw = {}

with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split()
        if len(parts) != embed_dim + 1:
            continue
        word = parts[0]
        glove_raw[word] = np.array(parts[1:], dtype=np.float32)

print(f"Loaded {len(glove_raw)} GloVe words")

# -----------------------------
# 3. Align GloVe to Brown vocab
# -----------------------------
vocab_size = len(word_to_index)
embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

for word, idx in word_to_index.items():
    if word == "<PAD>":
        embedding_matrix[idx] = np.zeros(embed_dim)
    elif word in glove_raw:
        embedding_matrix[idx] = glove_raw[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.01, size=(embed_dim,))

# -----------------------------
# 4. Normalize embeddings
# -----------------------------
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
embedding_matrix = embedding_matrix / norms

# -----------------------------
# 5. Save aligned GloVe
# -----------------------------
os.makedirs("embeddings", exist_ok=True)

torch.save(
    {
        "embeddings": torch.tensor(embedding_matrix, dtype=torch.float32),
        "word2idx": word_to_index,
        "idx2word": {i: w for w, i in word_to_index.items()}
    },
    "embeddings/glove.pt"
)

print("✅ Saved Brown-aligned GloVe embeddings to embeddings/glove.pt")