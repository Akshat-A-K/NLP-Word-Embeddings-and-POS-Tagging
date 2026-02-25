import nltk
from nltk.corpus import brown
import random
from collections import Counter
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


nltk.download('brown')
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

window_size = 2
print(f"Context window size: {window_size}")

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

X_train, y_train = create_context_window(train_sents, word_to_index, tag_to_index, window_size)
X_val, y_val = create_context_window(validation_sents, word_to_index, tag_to_index, window_size)
X_test, y_test = create_context_window(test_sents, word_to_index, tag_to_index, window_size)

print("Train examples:", len(X_train))
print("Validation examples:", len(X_val))
print("Test examples:", len(X_test))

X_train = torch.LongTensor(X_train)
y_train = torch.LongTensor(y_train)

X_val = torch.LongTensor(X_val)
y_val = torch.LongTensor(y_val)

X_test = torch.LongTensor(X_test)
y_test = torch.LongTensor(y_test)

print(X_train.shape)
print(y_train.shape)
print("Data preparation complete.")

class POSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = POSDataset(X_train, y_train)
val_dataset = POSDataset(X_val, y_val)
test_dataset = POSDataset(X_test, y_test)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.view(embedding.size(0), -1)

        out = self.fc1(embedding)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
embedding_paths = {
    "SVD": "embeddings/svd.pt",
    "Word2Vec": "embeddings/skipgram.pt",
    "GloVe": "embeddings/glove.pt"
}

results = {}
for name, path in embedding_paths.items():
    if os.path.exists(path):
        print(f"Loading {name} embeddings from {path}")
        embedding_data = torch.load(path)
        embedding_matrix = embedding_data['embeddings']
        embedding_matrix = embedding_matrix.to(device)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

        num_tags = len(unique_tags)
        model = MLPPosTagger(embedding_matrix, num_tags, window_size, freeze_embeddings=True).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_f1 = 0.0
        best_state = None
        epochs = 10

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = model.state_dict()

        model.load_state_dict(best_state)
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", cm)

        results[name] = {
            "accuracy": accuracy,
            "f1": f1,
            "y_true": all_labels,
            "y_pred": all_preds,
            "confusion_matrix": cm
        }

        print(f"{name} Test Accuracy: {accuracy:.4f}, "f"Macro-F1: {f1:.4f}")
    else:
        print(f"Embedding file {path} not found. Skipping.")

print("\n" + "="*60)
print("FINAL POS TAGGING RESULTS (TEST SET)")
print("="*60)

print(f"{'Embedding':<15} {'Accuracy':<12} {'Macro-F1':<12}")
print("-"*60)

for name, res in results.items():
    print(f"{name:<15} {res['accuracy']:<12.4f} {res['f1']:<12.4f}")
    print(f"Confusion Matrix for {name}:\n{res['confusion_matrix']}\n")

print("="*60)

best_model = max(results, key=lambda k: results[k]["f1"])
print(f"\nBest model: {best_model}")

cm = confusion_matrix(
    results[best_model]["y_true"],
    results[best_model]["y_pred"]
)

print("Confusion Matrix:")
print(cm)

for i, tag in enumerate(unique_tags):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    print(f"Tag: {tag}")
    print(f"  TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")