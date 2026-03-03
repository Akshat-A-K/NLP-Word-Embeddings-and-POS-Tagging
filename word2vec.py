import nltk
from nltk.corpus import brown
from collections import Counter
import numpy as np
import torch
import os
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target, context, negatives):
        v_w = self.embeddings(target)
        v_c = self.context_embeddings(context)
        v_n = self.context_embeddings(negatives)

        pos_score = torch.sum(v_w * v_c, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.bmm(v_n, v_w.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)

        return -torch.mean(pos_loss + neg_loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

param_grid = {
    "minimum_count": [2, 4],
    "embedding_dim": [100, 200, 300],
    "window_size": [2, 4],
    "num_negatives": [5, 10],
    "learning_rate": [0.001, 0.003],
    "epochs": [10],
    "batch_size": [256, 512]
}

best_score = -float("inf")
best_model = None
best_config = None

for minimum_count in param_grid["minimum_count"]:
    vocab = {word for word, count in word_counts.items() if count >= minimum_count}
    vocab = sorted(vocab)
    print(f"Vocabulary size after filtering, minimum_count={minimum_count}: {len(vocab)}")

    filtered_sentences = [[word for word in sent if word in vocab] for sent in sentences]
    print(f"Sentences with filtered vocab: {len(filtered_sentences)}")

    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    for window_size in param_grid["window_size"]:
        for embedding_dim in param_grid["embedding_dim"]:
            for num_negatives in param_grid["num_negatives"]:
                for learning_rate in param_grid["learning_rate"]:
                    for epochs in param_grid["epochs"]:
                        for batch_size in param_grid["batch_size"]:
                            print(f"Running: min_count={minimum_count}, win={window_size}, dim={embedding_dim}, neg={num_negatives}, lr={learning_rate}, epochs={epochs}, batch={batch_size}")
                            pairs = []
                            for sentence in filtered_sentences:
                                for i, target in enumerate(sentence):
                                    target_idx = word_to_index[target]
                                    start = max(0, i - window_size)
                                    end = min(len(sentence), i + window_size + 1)
                                    for j in range(start, end):
                                        if j != i:
                                            context = sentence[j]
                                            context_idx = word_to_index[context]
                                            pairs.append((target_idx, context_idx))

                            print(f"Total training pairs: {len(pairs)}")

                            word_frequency = np.array([word_counts[word] for word in vocab], dtype=np.float32)
                            unigram_distribution = word_frequency ** 0.75 # THis is negative sampling distribution
                            unigram_distribution /= unigram_distribution.sum()
                            print(f"Unigram distribution (first 10): {unigram_distribution[:10]}")

                            model = Word2Vec(len(vocab), embedding_dim).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                            pairs_array = np.array(pairs)
                            for epoch in range(epochs):
                                np.random.shuffle(pairs_array)
                                print(f"Epoch {epoch+1}/{epochs}")

                                epoch_loss = 0.0
                                num_batches = 0

                                batch_iter = tqdm(
                                    range(0, len(pairs_array), batch_size),
                                    desc=f"Epoch {epoch+1}"
                                )

                                for i in batch_iter:
                                    batch = pairs_array[i:i+batch_size]
                                    if len(batch) < batch_size:
                                        continue

                                    target_batch = torch.LongTensor(batch[:,0]).to(device)
                                    context_batch = torch.LongTensor(batch[:,1]).to(device)

                                    negatives_np = np.random.choice(
                                        len(unigram_distribution),
                                        size=(batch_size, num_negatives),
                                        p=unigram_distribution
                                    )
                                    negatives_batch = torch.LongTensor(negatives_np).to(device)

                                    optimizer.zero_grad()
                                    loss = model(target_batch, context_batch, negatives_batch)
                                    loss.backward()
                                    optimizer.step()

                                    epoch_loss += loss.item()
                                    num_batches += 1

                            if num_batches == 0:
                                continue
                            avg_epoch_loss = epoch_loss / num_batches
                            print(f"Average epoch loss: {avg_epoch_loss:.4f}")
                        
                            score = -avg_epoch_loss
                            if score > best_score:
                                best_score = score
                                best_config = (
                                    minimum_count,
                                    window_size,
                                    embedding_dim,
                                    num_negatives,
                                    learning_rate,
                                    epochs,
                                    batch_size
                                )

                                emb = model.embeddings.weight.detach().cpu()
                                emb = emb / emb.norm(p=2, dim=1, keepdim=True)

                                best_model = {
                                    'embeddings': emb.clone(),
                                    'word2idx': word_to_index.copy(),
                                    'idx2word': index_to_word.copy()
                                }


print("Best Skip-Gram configuration selected:")
print(best_config)

os.makedirs('embeddings', exist_ok=True)
torch.save(best_model, "embeddings/skipgram.pt")

print("Best Skip-Gram embeddings saved to embeddings/skipgram.pt")
