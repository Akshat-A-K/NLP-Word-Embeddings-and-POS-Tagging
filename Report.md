## Assignment 2 – Word Embeddings on the Brown Corpus

### 1. Task Overview

- **Goal**: Train and analyze word embeddings on the Brown corpus using:
  - **Frequency-based embeddings**: SVD over a PPMI co‑occurrence matrix.
  - **Prediction-based embeddings**: Word2Vec **Skip‑Gram with Negative Sampling**.
  - **Pre-trained embeddings**: Brown‑vocabulary‑aligned **GloVe 6B (300d)**.
- **Evaluation**:
  - Intrinsic tests: cosine similarities and nearest neighbors.
  - **Analogy test**: \(A : B :: C : ?\) for semantic/syntactic relationships (Task 2.1).
  - **Bias check**: cosine similarity between profession words and gender terms (Task 2.2).

All models use the **Brown corpus from NLTK** and the learned embeddings are saved as `.pt` files under `embeddings/`.

---

### 2. Data and Pre‑processing

- **Corpus**: `nltk.corpus.brown` (sentence-level tokenization).
- **Token filtering**:
  - Keep only **alphabetic tokens** (`word.isalpha()`).
  - Convert all tokens to **lowercase**.
  - Remove empty sentences after filtering.
- **Vocabulary construction**:
  - For **SVD**: vocabulary filtered by frequency threshold **min_count = 5**.
  - For **Skip‑Gram**: vocabulary filtered by **min_count = 2**.
  - These thresholds balance:
    - **Noise reduction** (dropping very rare words that are poorly estimated).
    - **Model size / runtime**, especially for the dense co‑occurrence matrix (SVD) and large number of training pairs (Skip‑Gram).

---

### 3. SVD Embeddings (Task 1.1)

#### 3.1 Co‑occurrence Matrix and Weighting

- **Context window**: symmetric window of size \(w\), i.e., for a focus word at position \(i\) we count words in \([i-w, i+w]\setminus\{i\}\).
- **Counts**: build a \(|V| \times |V|\) **word–context co‑occurrence matrix** \(C\) over the filtered sentences.
- **Probabilities**:
  - \(P(w,c) = \frac{C_{w,c}}{\sum_{w',c'} C_{w',c'}}\)
  - \(P(w) = \sum_c P(w,c)\), \(P(c) = \sum_w P(w,c)\)
- **PPMI weighting with smoothing**:
  - \(\text{PMI}(w,c) = \log \frac{P(w,c) + \epsilon}{P(w)P(c) + \epsilon}\)
  - \(\text{PPMI}(w,c) = \max(\text{PMI}(w,c), 0)\)
  - Small \(\epsilon\) avoids taking log of zero and stabilizes estimates for very rare events.

#### 3.2 Hyperparameter Search

- **Grid**:
  - `minimum_count`: \(\{2, 5\}\)
  - `window_size`: \(\{2, 3, 4\}\)
  - `embedding_dim`: \(\{100, 200, 300\}\)
  - `ppmi_smoothing`: \(\{1e{-8}, 1e{-6}, 1e{-4}\)\)
- For each configuration:
  - Build co‑occurrence matrix and PPMI.
  - Compute **matrix sparsity** (percentage of zero entries).
  - Apply **truncated SVD** (`scipy.sparse.linalg.svds`) with rank `embedding_dim`.
  - Construct word vectors as \(U S\) (sorted by descending singular value) and **L2‑normalize** rows.
- **Selection criterion**:
  - Target sparsity was set to about **98%**, a regime where:
    - The matrix is sparse enough to be memory‑efficient.
    - But still dense enough that PMI statistics are meaningful.
  - For each configuration, compute:
    - `score = |sparsity − 98.0|`
  - Choose the configuration with the **smallest score** (i.e., sparsity closest to 98%).

#### 3.3 Final SVD Configuration

- **Best configuration (from `svd_output.txt`)**:
  - `minimum_count = 5`
  - `window_size = 4`
  - `embedding_dim = 100`
  - `ppmi_smoothing = 1e-8`
  - Resulting sparsity ≈ **98.91%**
- **Justification**:
  - **Higher `minimum_count` (5)**:
    - Removes very rare words with unreliable co‑occurrence statistics.
    - Shrinks the vocabulary, directly reducing memory usage of the dense matrix.
  - **Larger window (`4`)**:
    - Captures broader topical/semantic contexts, which is beneficial for SVD‑style global embeddings.
    - Slightly reduces sparsity compared to smaller windows, bringing it closer to the 98% target.
  - **Moderate dimensionality (`100`)**:
    - Sufficient to encode major semantic axes for a relatively small corpus (Brown).
    - Lower dimensionality also acts as **regularization**, reducing overfitting to noise.
  - **Very small smoothing (`1e-8`)**:
    - Stabilizes PMI for low‑frequency pairs without excessively dampening informative PMI values.

The final SVD model is saved as **`embeddings/svd.pt`** (containing `embeddings`, `word2idx`, and `idx2word`).

---

### 4. Neural Word Embeddings – Skip‑Gram with Negative Sampling (Task 1.2)

#### 4.1 Model Architecture

- Implemented a **Word2Vec Skip‑Gram** model:
  - Two embedding matrices:
    - **Target embeddings**: `Embedding(vocab_size, embedding_dim)`
    - **Context embeddings**: `Embedding(vocab_size, embedding_dim)`
  - For each (target, context) pair:
    - Positive score: dot product \(v_w \cdot v_c\)
    - Negative samples: multiple context words drawn from a **noise distribution**.
  - **Loss** for each batch:
    - Maximize \(\log \sigma(v_w \cdot v_c)\) for positive pairs.
    - Maximize \(\sum_{n} \log \sigma(-v_w \cdot v_{n})\) for negative pairs.
    - Implemented as a **negative log‑likelihood** minimized with Adam.

#### 4.2 Negative Sampling and Training Data

- **Training pairs**:
  - Generated by sliding a symmetric window over filtered sentences
    (with `window_size` chosen during search; see below).
  - For each target word, all context words within the window yield a (target, context) pair.
- **Noise distribution**:
  - Based on token unigram counts \(f(w)\):
    - \(P_{\text{neg}}(w) \propto f(w)^{0.75}\)
  - This is the standard heuristic from Mikolov et al. that:
    - Downweights extremely frequent function words.
    - Still samples them enough to model syntactic patterns.

#### 4.3 Hyperparameter Search

- **Grid**:
  - `minimum_count`: \(\{2, 4\}\)
  - `embedding_dim`: \(\{100, 200, 300\}\)
  - `window_size`: \(\{2, 4\}\)
  - `num_negatives`: \(\{5, 10\}\)
  - `learning_rate`: \(\{0.001, 0.003\}\)
  - `epochs` (for search): \(\{10\}\)
  - `batch_size`: \(\{256, 512\}\)
- For each configuration:
  - Build vocabulary with the chosen `minimum_count`.
  - Construct all (target, context) pairs for the specified `window_size`.
  - Train Skip‑Gram with negative sampling for the given number of epochs.
  - Track the **average training loss** across batches.
- **Selection criterion**:
  - Use **negative average epoch loss** as the score:
    - `score = −avg_epoch_loss`
  - Choose the configuration with the **highest score** (i.e., lowest loss).

#### 4.4 Final Skip‑Gram Configuration

- **Best configuration (from `word2vec_skipgram.txt`)**:
  - `minimum_count = 2`
  - `window_size = 2`
  - `embedding_dim = 200`
  - `num_negatives = 5`
  - `learning_rate = 0.003`
  - `epochs (search stage) = 8`
  - `batch_size = 512`
  - Total training pairs ≈ **3.53M**
- After identifying the best config, the model was **retrained** for **15 final epochs** with the same hyperparameters, achieving a final average loss of ≈ **1.89**.
- **Justification**:
  - **Lower `minimum_count` (2)**:
    - Keeps more vocabulary coverage than SVD, which is important for downstream analogy tasks.
  - **Small `window_size` (2)**:
    - Encourages modeling of **local syntactic and functional relations** suitable for Skip‑Gram.
    - Reduces the number of training pairs and speeds up training.
  - **Intermediate dimension (200)**:
    - Large enough to represent rich semantics.
    - Still efficient to train on a relatively small corpus with millions of pairs.
  - **5 negative samples**:
    - Provides a good trade‑off between training stability and speed on this dataset.
  - **Higher learning rate (0.003) with Adam + batch_size 512**:
    - Empirically produced faster convergence and better (lower) loss than the more conservative 0.001.

The final Skip‑Gram embeddings are saved as **`embeddings/skipgram.pt`**.

---

### 5. Pre‑trained GloVe Embeddings

- **Source**: `glove.6B.300d.txt` (pre-trained on a large web/corpus mixture).
- **Alignment to Brown vocabulary** (see `convert_glove.py`):
  - Build a Brown vocabulary from the training portion of Brown POS‑tagged sentences:
    - Lowercase words.
    - Keep tokens with count ≥ 2.
    - Add a special `<PAD>` token.
  - For each Brown word:
    - If present in raw GloVe, copy its 300‑dimensional vector.
    - Otherwise, initialize a small **Gaussian random** vector.
  - L2‑normalize all rows to unit length.
- Save to **`embeddings/glove.pt`** with the same structure as the other models.

This allows a **fair comparison** between models on a common Brown‑derived vocabulary while benefiting from GloVe’s large‑corpus training.

---

### 6. Quantitative Comparison of Embeddings

All comparisons in this section are computed by `compare.py`, which:
- Loads `embeddings/svd.pt`, `embeddings/skipgram.pt`, and `embeddings/glove.pt`.
- Implements:
  - `cosine_sim(w1, w2)`
  - `most_similar(word, topk)`
  - `analogy(a, b, c, topk)`
- Prints tables for cosine similarity, nearest neighbors, analogies, and the bias check.

#### 6.1 Cosine Similarity of Word Pairs

Cosine similarity scores (from the console output you provided):

| Pair          | SVD    | Word2Vec | GloVe  |
|--------------|--------|----------|--------|
| jury – trial | 0.7001 | 0.2071   | 0.6365 |
| jury – said  | 0.2252 | 0.1792   | 0.1759 |
| the – a      | 0.7548 | 0.5544   | 0.5242 |
| court – case | 0.6811 | 0.2045   | 0.6455 |
| said – said  | 1.0000 | 1.0000   | 1.0000 |

- **Observations**:
  - For **semantically related legal terms** (e.g., `jury`–`trial`, `court`–`case`), **SVD and GloVe** assign **high similarity** (~0.64–0.70), while Skip‑Gram trained only on Brown gives much lower scores (~0.20).
  - For **function words** (`the`–`a`), all models give high similarity, with SVD highest (~0.75).
  - The GloVe model matches intuitive semantic similarity best overall, reflecting its training on a much larger corpus.

#### 6.2 Nearest Neighbors (Most Similar Words)

Example neighbors for selected query words:

- **`trial`**:
  - **SVD**: `jury`, `testimony`, `investigation`, `court`, `election` (legal and procedural terms).
  - **GloVe**: `prosecution`, `trials`, `defendants`, `case`, `prosecutors` (very coherent legal cluster).
  - **Skip‑Gram**: neighbors such as `subtype`, `commencement`, `crus`, `repeat`, `harvests`, which are **less semantically focused** and often noisy.
- **`jury`**:
  - **SVD** and **GloVe** again retrieve **legal concepts** (`investigation`, `witnesses`, `trial`, `verdict`, `judge`).
  - **Skip‑Gram** includes more idiosyncratic or weakly related words (`thanking`, `stranger`, `embark`).

Overall, **SVD and GloVe** capture coherent semantic neighborhoods around frequent words, whereas the **Brown‑trained Skip‑Gram** embeddings appear noisier due to limited data and smaller effective context.

---

### 7. Analogy Test – Semantic Capability (Task 2.1)

The standard analogy formula is used:
\[
\arg\max_x \cos\big(x,\; \vec{B} - \vec{A} + \vec{C}\big)
\]

For each of the three embedding variants, the **top 5 predicted words** are:

#### 7.1 Paris : France :: Delhi : ?

| Model   | Top‑5 predictions (word, cosine)                                                |
|---------|----------------------------------------------------------------------------------|
| **SVD** | nato (0.58), australia (0.55), auspices (0.54), alliance (0.50), treaty (0.50) |
| **Skip‑Gram** | deficit (0.32), fha (0.32), roundup (0.31), procurer (0.31), sentry (0.30) |
| **GloVe** | india (0.73), pakistan (0.62), indian (0.52), australia (0.40), hindu (0.40) |

- **Analysis**:
  - Only **GloVe** correctly links `Delhi` to **`india`** at the top of the list and clusters around regional/ethnic terms.
  - SVD associates `Delhi` with **international organizations and alliances**, which is thematically related to geopolitics but not the specific capital–country relation.
  - Skip‑Gram fails to capture the geographic relation, producing mostly unrelated terms.

#### 7.2 King : Man :: Queen : ?

| Model   | Top‑5 predictions (word, cosine)                                              |
|---------|------------------------------------------------------------------------------|
| **SVD** | woman (0.62), gentle (0.59), young (0.57), boy (0.54), girl (0.52)          |
| **Skip‑Gram** | bashful (0.34), friend (0.32), gouged (0.31), like (0.30), undisputed (0.30) |
| **GloVe** | woman (0.70), girl (0.56), person (0.51), she (0.48), mother (0.46)       |

- **Analysis**:
  - Both **SVD** and **GloVe** correctly move along a **gender direction**: from `man` to `woman`.
  - GloVe’s top answer `woman` matches the expected semantic relation best.
  - Skip‑Gram again produces largely irrelevant or weakly related words, showing that the embedding geometry is not robust enough for this type of analogy.

#### 7.3 Swim : Swimming :: Run : ?

| Model   | Top‑5 predictions (word, cosine)                                                |
|---------|----------------------------------------------------------------------------------|
| **SVD** | going (0.63), ran (0.63), straight (0.62), move (0.61), look (0.61)           |
| **Skip‑Gram** | asses (0.33), rooster (0.32), smoothing (0.32), stalking (0.31), barking (0.31) |
| **GloVe** | running (0.57), runs (0.54), three (0.47), ran (0.47), two (0.46)           |

- **Analysis**:
  - **GloVe** captures the **inflectional/tense pattern** well: `run` → `running` / `runs` mirrors `swim` → `swimming`.
  - SVD detects some loose motion/verb semantics but not the exact morphological mapping.
  - Skip‑Gram fails quite dramatically, giving unrelated nouns and noisy verbs.

**Conclusion for Task 2.1**:  
The **pre‑trained GloVe embeddings** clearly provide the strongest semantic and morphological structure.  
SVD embeddings capture some semantic regularities but are weaker in analogy structure.  
Skip‑Gram trained only on Brown performs the worst on analogies, mainly due to limited corpus size and training signal.

---

### 8. Bias Check – Ethical Capability (Task 2.2, GloVe only)

For the **pre‑trained GloVe** embeddings we compute:

- Pair A: `cos(vec('doctor'), vec('man'))` vs. `cos(vec('doctor'), vec('woman'))`
- Pair B: `cos(vec('nurse'), vec('man'))` vs. `cos(vec('nurse'), vec('woman'))`
- Pair C: `cos(vec('homemaker'), vec('man'))` vs. `cos(vec('homemaker'), vec('woman'))`

From `compare.py` the similarities are:

| Profession | man    | woman  |
|------------|--------|--------|
| doctor     | 0.4012 | 0.4691 |
| nurse      | 0.2373 | 0.4496 |
| homemaker  | 0.0529 | 0.2857 |

- **Observations**:
  - For all three professions, the embedding of the profession is **more similar to `woman` than to `man`**.
  - The effect is especially strong for **`nurse`** and **`homemaker`**, which are much closer to `woman` than to `man`.
  - Even for **`doctor`**, which we might hope to be neutral, the vector is **tilted towards `woman`** in this particular GloVe slice.
- **Interpretation**:
  - The GloVe space clearly reflects **gender stereotypes** present in its training data:
    - Caregiving and domestic roles (`nurse`, `homemaker`) are encoded as more female.
  - This confirms the findings of Bolukbasi et al. (2016) that **word embeddings encode harmful social biases**.

These biases would **propagate to downstream models** (e.g., resume screening, search ranking) unless explicitly mitigated (e.g., via debiasing algorithms, balanced training data, or constrained representations).

---

### 9. Overall Discussion – Are the Embeddings Fishy?

- **SVD (Brown‑trained)**:
  - Captures reasonable **semantic similarity** among frequent words (especially within domains like law in the Brown corpus).
  - Performs moderately on analogy tasks but is limited by:
    - The relatively small size and domain coverage of Brown.
    - The global, count‑based nature of SVD, which is less tuned to fine‑grained analogy geometry.
- **Skip‑Gram (Brown‑trained)**:
  - In principle, should model local syntactic and semantic relations well.
  - In practice, on this small corpus, the resulting space is **noisy**:
    - Nearest neighbors are often unintuitive.
    - Analogy performance is poor, suggesting underfitting / insufficient data.
- **Pre‑trained GloVe**:
  - Exhibits **strong semantic, syntactic, and morphological structure**, clearly outperforming Brown‑only models on analogies and nearest neighbors.
  - However, it embeds **real‑world biases**, especially gender stereotypes relating to professions.

**Answer to “Are the embeddings fishy?”**:
- From a **semantic capability** perspective, pre‑trained GloVe is the most powerful, SVD is decent, and Brown Skip‑Gram is weakest.
- From an **ethical capability** perspective, GloVe is clearly **“fishy”**: it encodes stereotypical gender associations that must be audited and mitigated before deployment.

---

### 10. Files Produced

- **Training scripts**:
  - `svd_embeddings.py` – trains and saves `embeddings/svd.pt`.
  - `word2vec_skipgram.py` – trains and saves `embeddings/skipgram.pt`.
  - `convert_glove.py` – aligns GloVe and saves `embeddings/glove.pt`.
- **Analysis script**:
  - `compare.py` – runs cosine similarity, nearest neighbors, analogy tests, and the bias check; produces the console output summarized above.
- **Embedding files**:
  - `embeddings/svd.pt`
  - `embeddings/skipgram.pt`
  - `embeddings/glove.pt`

This report summarizes the training choices, evaluation protocol, and key findings required for Assignment 2, Tasks 1 and 2.

