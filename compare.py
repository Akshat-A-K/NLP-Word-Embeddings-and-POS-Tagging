import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


embedding_files = [
    ("SVD", 'embeddings/svd.pt'),
    ("Word2Vec", 'embeddings/skipgram.pt'),
    ("GloVe", 'embeddings/glove.pt'),
]

results = {
    'cosine': {},
    'most_similar': {},
    'analogy': {},
}

test_pairs = [
    ('jury', 'trial'),
    ('jury', 'said'),
    ('the', 'a'),
    ('court', 'case'),
    ('said', 'said'),
]
test_words = ['trial', 'jury', 'said', 'court', 'case']
analogy_questions = [
    ('man', 'woman', 'king'),
    ('paris', 'france', 'london'),
    ('jury', 'trial', 'court'),
    ('said', 'asked', 'told'),
]

for emb_name, emb_path in embedding_files:
    print(f"\nLoading {emb_name} embeddings from {emb_path}...")
    embeddings_data = torch.load(emb_path)
    embeddings = embeddings_data['embeddings'].numpy()
    word_to_index = embeddings_data['word2idx']
    index_to_word = embeddings_data['idx2word']

    def cosine_sim(w1, w2):
        if w1 not in word_to_index or w2 not in word_to_index:
            return None
        idx1 = word_to_index[w1]
        idx2 = word_to_index[w2]
        emb2 = embeddings[idx2].reshape(1, -1)
        emb1 = embeddings[idx1].reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)[0][0]
        return sim

    def most_similar(word, topk=5):
        if word not in word_to_index:
            return []
        word_idx = word_to_index[word]
        word_emb = embeddings[word_idx].reshape(1, -1)
        similarities = cosine_similarity(word_emb, embeddings)[0]
        top_indices = np.argsort(-similarities)[:topk + 1]
        results = []
        for idx in top_indices:
            if idx != word_idx:
                results.append((index_to_word[idx], float(similarities[idx])))
            if len(results) == topk:
                break
        return results

    def analogy(a, b, c, topk=1):
        if a not in word_to_index or b not in word_to_index or c not in word_to_index:
            return ["NOT IN VOCAB"]
        a_emb = embeddings[word_to_index[a]]
        b_emb = embeddings[word_to_index[b]]
        c_emb = embeddings[word_to_index[c]]
        target_emb = (b_emb - a_emb + c_emb).reshape(1, -1)
        similarities = cosine_similarity(target_emb, embeddings)[0]
        top_indices = np.argsort(-similarities)[:topk + 3]
        results = []
        exclude_indices = {word_to_index[a], word_to_index[b], word_to_index[c]}
        for idx in top_indices:
            if idx not in exclude_indices:
                results.append(index_to_word[idx])
            if len(results) == topk:
                break
        return results

    # Cosine similarity
    results['cosine'][emb_name] = []
    for w1, w2 in test_pairs:
        sim = cosine_sim(w1, w2)
        results['cosine'][emb_name].append((w1, w2, sim))

    # Most similar
    results['most_similar'][emb_name] = {}
    for word in test_words:
        results['most_similar'][emb_name][word] = most_similar(word, topk=5)

    # Analogy
    results['analogy'][emb_name] = {}
    for a, b, c in analogy_questions:
        ans = analogy(a, b, c, topk=1)[0]
        results['analogy'][emb_name][f"{a}:{b}::{c}:?"] = ans

# Print results in tables

# Print results in tables (now includes GloVe)
print("\n==================== COSINE SIMILARITY RESULTS ====================")
print(f"{'Pair':<20} {'SVD':<12} {'Word2Vec':<12} {'GloVe':<12}")
print("-" * 60)
for idx in range(len(test_pairs)):
    w1, w2, svd_sim = results['cosine']['SVD'][idx]
    _, _, w2v_sim = results['cosine']['Word2Vec'][idx]
    _, _, glove_sim = results['cosine']['GloVe'][idx]
    svd_str = f"{svd_sim:.4f}" if svd_sim is not None else "NOT IN VOCAB"
    w2v_str = f"{w2v_sim:.4f}" if w2v_sim is not None else "NOT IN VOCAB"
    glove_str = f"{glove_sim:.4f}" if glove_sim is not None else "NOT IN VOCAB"
    print(f"{w1} - {w2:<15} {svd_str:<12} {w2v_str:<12} {glove_str:<12}")
print("-" * 60)

print("\n==================== MOST SIMILAR WORDS ====================")
print(f"{'Query':<12} {'SVD':<40} {'Word2Vec':<40} {'GloVe':<40}")
print("-" * 130)
for word in test_words:
    svd_sim = ', '.join([f"{w} ({s:.2f})" for w, s in results['most_similar']['SVD'][word]])
    w2v_sim = ', '.join([f"{w} ({s:.2f})" for w, s in results['most_similar']['Word2Vec'][word]])
    glove_sim = ', '.join([f"{w} ({s:.2f})" for w, s in results['most_similar']['GloVe'][word]])
    print(f"{word:<12} {svd_sim:<40} {w2v_sim:<40} {glove_sim:<40}")
print("-" * 130)

print("\n==================== ANALOGY RESULTS ====================")
print(f"{'Analogy':<35} {'SVD':<20} {'Word2Vec':<20} {'GloVe':<20}")
print("-" * 95)
for key in results['analogy']['SVD'].keys():
    svd_ans = results['analogy']['SVD'][key]
    w2v_ans = results['analogy']['Word2Vec'][key]
    glove_ans = results['analogy']['GloVe'][key]
    print(f"{key:<35} {svd_ans:<20} {w2v_ans:<20} {glove_ans:<20}")
print("-" * 95)

# Task 2.1: Analogy Test (Top 5 for given examples)
print("\n==================== ANALOGY TASK 2.1: TOP 5 WORDS ====================")
analogy_examples = [
    ("paris", "france", "delhi"),
    ("king", "man", "queen"),
    ("swim", "swimming", "run"),
]
analogy_labels = [
    "Paris : France :: Delhi : ?",
    "King : Man :: Queen : ?",
    "Swim : Swimming :: Run : ?"
]
for i, (a, b, c) in enumerate(analogy_examples):
    print(f"\n{analogy_labels[i]}")
    for emb_name in ["SVD", "Word2Vec", "GloVe"]:
        # Use the correct embedding context
        embeddings_data = torch.load(embedding_files[[x[0] for x in embedding_files].index(emb_name)][1])
        embeddings = embeddings_data['embeddings'].numpy()
        word_to_index = embeddings_data['word2idx']
        index_to_word = embeddings_data['idx2word']
        def analogy_topk(a, b, c, topk=5):
            if a not in word_to_index or b not in word_to_index or c not in word_to_index:
                return ["NOT IN VOCAB"]
            a_emb = embeddings[word_to_index[a]]
            b_emb = embeddings[word_to_index[b]]
            c_emb = embeddings[word_to_index[c]]
            target_emb = (b_emb - a_emb + c_emb).reshape(1, -1)
            similarities = cosine_similarity(target_emb, embeddings)[0]
            top_indices = np.argsort(-similarities)[:topk + 3]
            results = []
            exclude_indices = {word_to_index[a], word_to_index[b], word_to_index[c]}
            for idx in top_indices:
                if idx not in exclude_indices:
                    results.append((index_to_word[idx], float(similarities[idx])))
                if len(results) == topk:
                    break
            return results
        top5 = analogy_topk(a, b, c, topk=5)
        top5_str = ', '.join([f"{w} ({s:.2f})" for w, s in top5])
        print(f"  {emb_name:<8}: {top5_str}")

# Task 2.2: Bias Check (GloVe only)
print("\n==================== BIAS CHECK (GloVe only) ====================")
glove_data = torch.load(embedding_files[[x[0] for x in embedding_files].index("GloVe")][1])
glove_emb = glove_data['embeddings'].numpy()
glove_word2idx = glove_data['word2idx']
def glove_cosine(w1, w2):
    if w1 not in glove_word2idx or w2 not in glove_word2idx:
        return None
    idx1 = glove_word2idx[w1]
    idx2 = glove_word2idx[w2]
    emb1 = glove_emb[idx1].reshape(1, -1)
    emb2 = glove_emb[idx2].reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

professions = ["doctor", "nurse", "homemaker"]
pairs = [("man", "woman")]
for prof in professions:
    for g1, g2 in pairs:
        sim1 = glove_cosine(prof, g1)
        sim2 = glove_cosine(prof, g2)
        print(f"{prof.capitalize()} - {g1}: {sim1:.4f} | {prof.capitalize()} - {g2}: {sim2:.4f}")


def cosine_sim(w1, w2):
    if w1 not in word_to_index or w2 not in word_to_index:
        return None
    
    idx1 = word_to_index[w1]
    idx2 = word_to_index[w2]
    
    emb2 = embeddings[idx2].reshape(1, -1)
    emb1 = embeddings[idx1].reshape(1, -1)
    
    sim = cosine_similarity(emb1, emb2)[0][0]
    return sim


def most_similar(word, topk=5):
    if word not in word_to_index:
        return []
    
    word_idx = word_to_index[word]
    word_emb = embeddings[word_idx].reshape(1, -1)
    similarities = cosine_similarity(word_emb, embeddings)[0]
    top_indices = np.argsort(-similarities)[:topk + 1]
    results = []
    for idx in top_indices:
        if idx != word_idx:
            results.append((index_to_word[idx], float(similarities[idx])))
        if len(results) == topk:
            break
    return results


def analogy(a, b, c, topk=5):
    if a not in word_to_index or b not in word_to_index or c not in word_to_index:
        return []
    
    a_emb = embeddings[word_to_index[a]]
    b_emb = embeddings[word_to_index[b]]
    c_emb = embeddings[word_to_index[c]]
    target_emb = (b_emb - a_emb + c_emb).reshape(1, -1) 
    similarities = cosine_similarity(target_emb, embeddings)[0]
    top_indices = np.argsort(-similarities)[:topk + 3]
    results = []
    exclude_indices = {word_to_index[a], word_to_index[b], word_to_index[c]}
    for idx in top_indices:
        if idx not in exclude_indices:
            results.append((index_to_word[idx], float(similarities[idx])))
        if len(results) == topk:
            break
    
    return results

print("=" * 60)
print("COMPREHENSIVE FUNCTION TESTS")
print("=" * 60)
print("\n--- Test 1: cosine_sim(w1, w2) ---")
print("Testing cosine similarity between word pairs:")
test_pairs = [
    ('jury', 'trial'),
    ('jury', 'said'),
    ('the', 'a'),
    ('court', 'case'),
    ('said', 'said')
]

for w1, w2 in test_pairs:
    sim = cosine_sim(w1, w2)
    if sim is not None:
        print(f"  cosine_sim('{w1}', '{w2}') = {sim:.4f}")
    else:
        print(f"  cosine_sim('{w1}', '{w2}') = NOT IN VOCAB")
print("\n--- Test 2: most_similar(word, topk=5) ---")
test_words = ['trial', 'jury', 'said', 'court', 'case']

for word in test_words:
    print(f"\nMost similar to '{word}':")
    similar = most_similar(word, topk=5)
    for i, (sim_word, sim_score) in enumerate(similar, 1):
        print(f"  {sim_word}: {sim_score:.4f}")
print("\n--- Test 3: most_similar(word, topk) with different k values ---")
word = 'trial'
for k in [3, 5, 10]:
    print(f"\nTop {k} most similar to '{word}':")
    similar = most_similar(word, topk=k)
    for i, (sim_word, sim_score) in enumerate(similar, 1):
        print(f"  {sim_word}: {sim_score:.4f}")

print("\n--- Test 4: analogy(a, b, c, topk=5) ---")
test_analogies = [
    ('jury', 'trial', 'court'),
    ('said', 'asked', 'told')
]

for a, b, c in test_analogies:
    print(f"\nAnalogy: {a}:{b} :: {c}:?")
    results = analogy(a, b, c, topk=5)
    for i, (word, score) in enumerate(results, 1):
        print(f"  {word}: {score:.4f}")

print("\n--- Test 5: analogy(a, b, c, topk) with different k values ---")
a, b, c = 'jury', 'trial', 'court'
for k in [3, 5, 10]:
    print(f"\nAnalogy: {a}:{b} :: {c}:? (Top {k})")
    results = analogy(a, b, c, topk=k)
    for i, (word, score) in enumerate(results, 1):
        print(f"  {word}: {score:.4f}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
