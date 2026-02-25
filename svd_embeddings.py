import nltk
from nltk.corpus import brown
import numpy as np
from scipy.sparse.linalg import svds
import torch
import os
from collections import Counter

nltk.download('brown')

sentences = brown.sents()
print(f"Total sentences: {len(sentences)}")

def is_valid_word(word):
    return word.isalpha()

sentences = [[word.lower() for word in sent if is_valid_word(word)] for sent in sentences]
sentences = [sent for sent in sentences if len(sent) > 0]

print(f"Sentences after cleaning: {len(sentences)}")

all_words = [word for sentence in sentences for word in sentence]
print(f"Total tokens after cleaning: {len(all_words)}")

print("\nPreprocessing Filtering vocabulary")

word_counts = Counter(all_words)
print(f"Unique word types before filtering: {len(word_counts)}")

param_grid = {
    "minimum_count": [2, 5],
    "window_size": [2, 3, 4],
    "embedding_dim": [100, 200, 300],
    "ppmi_smoothing": [1e-8, 1e-6, 1e-4]
}

results = []
best_score = float("inf")
best_model = None
best_config = None

with open("output.txt", "w") as log:
    log.write("SVD Hyperparameter Tuning Results\n")
    for min_count in param_grid["minimum_count"]:
        for win in param_grid["window_size"]:
            for dim in param_grid["embedding_dim"]:
                for smooth in param_grid["ppmi_smoothing"]:
                    print(f"Running: min={min_count}, win={win}, dim={dim}, smooth={smooth}")
                    vocab = {w for w, c in word_counts.items() if c >= min_count}
                    vocab = sorted(vocab)

                    word_to_index = {w: i for i, w in enumerate(vocab)}
                    index_to_word = {i: w for w, i in word_to_index.items()}
                    
                    filtered_sentences = [
                        [w for w in sent if w in word_to_index]
                        for sent in sentences
                    ]

                    cooccurrence_matrix = np.zeros(
                        (len(vocab), len(vocab)), dtype=np.float32
                    )

                    for sentence in filtered_sentences:
                        for i in range(len(sentence)):
                            wi = word_to_index[sentence[i]]
                            for j in range(max(0, i - win), min(len(sentence), i + win + 1)):
                                if i != j:
                                    wj = word_to_index[sentence[j]]
                                    cooccurrence_matrix[wi][wj] += 1

                    total = cooccurrence_matrix.sum()
                    pwc = cooccurrence_matrix / total
                    pw = cooccurrence_matrix.sum(axis=1, keepdims=True) / total
                    pc = cooccurrence_matrix.sum(axis=0, keepdims=True) / total

                    pmi = np.log((pwc + smooth) / (pw * pc + smooth))
                    pmi = np.maximum(pmi, 0)

                    U, S, _ = svds(pmi, k=dim)
                    idx = np.argsort(S)[::-1]
                    embeddings = U[:, idx] * S[idx]

                    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10

                    sparsity = (1 - np.count_nonzero(cooccurrence_matrix) / cooccurrence_matrix.size) * 100

                    target_sparsity = 98.0
                    score = abs(sparsity - target_sparsity)

                    if score < best_score:
                        best_score = score
                        best_config = (min_count, win, dim, smooth, sparsity)
                        best_model = {
                            'embeddings': torch.FloatTensor(embeddings.copy()),
                            'word2idx': word_to_index.copy(),
                            'idx2word': index_to_word.copy()
                        }
                    
                    log.write(f"min={min_count}, win={win}, dim={dim}, smooth={smooth}, sparsity={sparsity:.2f}\n")

print("Best configuration selected:")
print(best_config)

os.makedirs('embeddings', exist_ok=True)
torch.save(best_model, "embeddings/svd.pt")
print("Final embeddings saved to embeddings/svd.pt")
