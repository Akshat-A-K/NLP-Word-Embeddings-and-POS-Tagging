import nltk
from nltk.corpus import brown
import random
from collections import Counter
import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

nltk.download('brown', quiet=True)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tagged_sents = list(brown.tagged_sents(tagset='universal'))
print(f"Total tagged sentences: {len(tagged_sents)}")

random.shuffle(tagged_sents)

num_sentences = len(tagged_sents)
train_sents = tagged_sents[:int(0.8 * num_sentences)]
validation_sents = tagged_sents[int(0.8 * num_sentences):int(0.9 * num_sentences)]
test_sents = tagged_sents[int(0.9 * num_sentences):]

print(f"Training sentences: {len(train_sents)}")
print(f"Validation sentences: {len(validation_sents)}")
print(f"Test sentences: {len(test_sents)}")

all_words = []
all_tags = []
for sent in train_sents:
    for word, tag in sent:
        all_words.append(word.lower())
        all_tags.append(tag)

word_counts = Counter(all_words)
vocab = {word for word, count in word_counts.items() if count >= 2}
vocab = sorted(vocab)
vocab = ['<PAD>'] + vocab

word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

unique_tags = sorted(set(all_tags))
tag_to_index = {tag: idx for idx, tag in enumerate(unique_tags)}
index_to_tag = {idx: tag for tag, idx in tag_to_index.items()}

print(f"Vocabulary size: {len(vocab)}")
print(f"Tag set size: {len(unique_tags)}")
print(f"Tags: {unique_tags}")


def create_context_window(sentences, word_to_index, tag_to_index, window_size):
    X = []
    y = []

    pad_idx = word_to_index['<PAD>']
    for sentence in sentences:
        words = [word.lower() for word, tag in sentence]
        tags = [tag for word, tag in sentence]

        indexed_words = [word_to_index.get(word, pad_idx) for word in words]

        for i in range(len(words)):
            context = []
            for j in range(i - window_size, i + window_size + 1):
                if j < 0 or j >= len(words):
                    context.append(pad_idx)
                else:
                    context.append(indexed_words[j])
            
            X.append(context)
            y.append(tag_to_index[tags[i]])
    
    return X, y


class POSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPPosTagger(nn.Module):
    def __init__(self, embedding_matrix, num_tags, window_size, hidden_dim=256, freeze_embeddings=True):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)

        if freeze_embeddings:
            self.embedding.weight.requires_grad_ = False
        
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.input_dim = (2 * window_size + 1) * embedding_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_tags)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.view(embedding.size(0), -1)

        out = self.relu(self.fc1(embedding))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def align_embeddings_to_vocab(embedding_data, word_to_index, device):
    loaded_emb = embedding_data['embeddings']
    if isinstance(loaded_emb, np.ndarray):
        loaded_emb = torch.FloatTensor(loaded_emb)
    loaded_word2idx = embedding_data['word2idx']
    _, dim = loaded_emb.shape
    aligned = torch.zeros(len(word_to_index), dim, dtype=torch.float32)
    for word, idx in word_to_index.items():
        if word == '<PAD>':
            aligned[idx] = 0.0
        elif word in loaded_word2idx:
            aligned[idx] = loaded_emb[loaded_word2idx[word]].clone()
        else:
            aligned[idx] = torch.randn(dim) * 0.01
    aligned[0] = 0.0
    return aligned.to(device)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm, all_labels, all_preds

embedding_dir = "/kaggle/input/datasets/akshat933/embeddings" if os.path.exists("/kaggle") else "embeddings"
embedding_paths = {
    "SVD": os.path.join(embedding_dir, "svd.pt"),
    "Word2Vec": os.path.join(embedding_dir, "skipgram.pt"),
    "GloVe": os.path.join(embedding_dir, "glove.pt"),
}

param_grid = {
    "window_size": [2, 4],
    "hidden_dim": [256, 512],
    "batch_size": [128, 256, 512],
    "learning_rate": [0.001, 0.003],
    "epochs": [20],
}

results_log = []
best_val_f1 = -1.0
best_config = None
best_model_state = None
best_embedding_name = None

print_every = max(1, int(os.environ.get("PRINT_EVERY", "20")))
total_configs = (
    len(embedding_paths)
    * len(param_grid["window_size"])
    * len(param_grid["hidden_dim"])
    * len(param_grid["batch_size"])
    * len(param_grid["learning_rate"])
    * len(param_grid["epochs"])
)
total_progress_lines = math.ceil(total_configs / print_every)
print(
    f"Grid search configs: {total_configs}. "
    f"Console updates every {print_every} configs (~{total_progress_lines} lines)."
)
config_counter = 0

print("Pre-computing context windows for all window sizes...")
window_cache = {}
for ws in param_grid["window_size"]:
    X_tr, y_tr = create_context_window(train_sents, word_to_index, tag_to_index, ws)
    X_vl, y_vl = create_context_window(validation_sents, word_to_index, tag_to_index, ws)
    X_te, y_te = create_context_window(test_sents, word_to_index, tag_to_index, ws)
    window_cache[ws] = (
        torch.LongTensor(X_tr), torch.LongTensor(y_tr),
        torch.LongTensor(X_vl), torch.LongTensor(y_vl),
        torch.LongTensor(X_te), torch.LongTensor(y_te),
    )
print("Context windows cached.")

with open("pos_tagger_results.txt", "w") as log:
    log.write("POS Tagger Grid Search Results\n")
    log.write("=" * 80 + "\n")

    for emb_name, emb_path in embedding_paths.items():
        if not os.path.exists(emb_path):
            log.write(f"Embedding file {emb_path} not found. Skipping {emb_name}.\n")
            continue

        embedding_data = torch.load(emb_path, map_location=device)
        embedding_matrix = align_embeddings_to_vocab(embedding_data, word_to_index, device)

        for window_size in param_grid["window_size"]:
            X_train, y_train, X_val, y_val, X_test, y_test = window_cache[window_size]

            for hidden_dim in param_grid["hidden_dim"]:
                for batch_size in param_grid["batch_size"]:
                    for lr in param_grid["learning_rate"]:
                        for epochs in param_grid["epochs"]:
                            config_str = f"emb={emb_name}, win={window_size}, hid={hidden_dim}, bs={batch_size}, lr={lr}, ep={epochs}"
                            config_counter += 1
                            if (
                                config_counter == 1
                                or config_counter % print_every == 0
                                or config_counter == total_configs
                            ):
                                print(f"Progress {config_counter}/{total_configs}: {config_str}")

                            train_dataset = POSDataset(X_train, y_train)
                            val_dataset = POSDataset(X_val, y_val)
                            test_dataset = POSDataset(X_test, y_test)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size)

                            num_tags = len(unique_tags)
                            model = MLPPosTagger(
                                embedding_matrix, num_tags, window_size,
                                hidden_dim=hidden_dim, freeze_embeddings=True
                            ).to(device)

                            criterion = nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            best_val_f1_run = -1.0
                            best_state_run = None


                            patience = 6
                            patience_counter = 0
                            last_best_val_f1 = -1.0
                            for epoch in range(1, epochs + 1):
                                model.train()
                                total_loss = 0.0
                                for X_batch, y_batch in train_loader:
                                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                    optimizer.zero_grad()
                                    outputs = model(X_batch)
                                    loss = criterion(outputs, y_batch)
                                    loss.backward()
                                    optimizer.step()
                                    total_loss += loss.item()

                                val_acc, val_f1, _, _, _ = evaluate(model, val_loader, device)

                                if val_f1 > best_val_f1_run:
                                    best_val_f1_run = val_f1
                                    best_state_run = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                                    patience_counter = 0
                                else:
                                    patience_counter += 1

                                if patience_counter >= patience:
                                    print(f"Early stopping at epoch {epoch} for config: {config_str}")
                                    break

                            model.load_state_dict(best_state_run)
                            model.to(device)

                            train_acc, train_f1, train_cm, _, _ = evaluate(model, train_loader, device)
                            val_acc, val_f1, val_cm, _, _ = evaluate(model, val_loader, device)
                            test_acc, test_f1, test_cm, y_true, y_pred = evaluate(model, test_loader, device)

                            row = {
                                "embedding": emb_name,
                                "window_size": window_size,
                                "hidden_dim": hidden_dim,
                                "batch_size": batch_size,
                                "lr": lr,
                                "epochs": epochs,
                                "train_acc": train_acc,
                                "train_f1": train_f1,
                                "val_acc": val_acc,
                                "val_f1": val_f1,
                                "test_acc": test_acc,
                                "test_f1": test_f1,
                                "test_cm": test_cm,
                                "y_true": y_true,
                                "y_pred": y_pred,
                            }
                            results_log.append(row)

                            log.write(f"\n{config_str}\n")
                            log.write(f"  Train  - Accuracy: {train_acc:.4f}  Macro-F1: {train_f1:.4f}\n")
                            log.write(f"  Val    - Accuracy: {val_acc:.4f}  Macro-F1: {val_f1:.4f}\n")
                            log.write(f"  Test   - Accuracy: {test_acc:.4f}  Macro-F1: {test_f1:.4f}\n")

                            if val_f1 > best_val_f1:
                                best_val_f1 = val_f1
                                best_config = config_str
                                best_model_state = row
                                best_embedding_name = emb_name

    log.write("\n" + "=" * 80 + "\n")
    log.write("BEST CONFIGURATION (by validation Macro-F1)\n")
    log.write("=" * 80 + "\n")
    if best_config:
        log.write(f"{best_config}\n")
        log.write(f"  Train  - Accuracy: {best_model_state['train_acc']:.4f}  Macro-F1: {best_model_state['train_f1']:.4f}\n")
        log.write(f"  Val    - Accuracy: {best_model_state['val_acc']:.4f}  Macro-F1: {best_model_state['val_f1']:.4f}\n")
        log.write(f"  Test   - Accuracy: {best_model_state['test_acc']:.4f}  Macro-F1: {best_model_state['test_f1']:.4f}\n")
        log.write(f"\nTest Confusion Matrix:\n{best_model_state['test_cm']}\n")
    log.write("\n" + "=" * 80 + "\n")
    log.write("SUMMARY TABLE\n")
    log.write("=" * 80 + "\n")
    log.write(f"{'Embedding':<10} {'Win':<4} {'Hid':<4} {'BS':<5} {'LR':<8} {'Epochs':<6} {'Train Acc':<10} {'Train F1':<10} {'Val Acc':<10} {'Val F1':<10} {'Test Acc':<10} {'Test F1':<10}\n")
    log.write("-" * 110 + "\n")
    for r in results_log:
        log.write(f"{r['embedding']:<10} {r['window_size']:<4} {r['hidden_dim']:<4} {r['batch_size']:<5} {r['lr']:<8} {r['epochs']:<6} {r['train_acc']:<10.4f} {r['train_f1']:<10.4f} {r['val_acc']:<10.4f} {r['val_f1']:<10.4f} {r['test_acc']:<10.4f} {r['test_f1']:<10.4f}\n")

print(f"\nResults saved to pos_tagger_results.txt")

if best_config:
    print(f"Best config: {best_config}")
    print(f"Test Accuracy: {best_model_state['test_acc']:.4f}, Test Macro-F1: {best_model_state['test_f1']:.4f}")
