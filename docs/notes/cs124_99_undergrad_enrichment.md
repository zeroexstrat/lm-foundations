# CS124 Source Extraction For Notebook 99 Undergrad Enrichment

This note extracts undergraduate-relevant material from the local CS124 source bundle at
`/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124`.

The purpose is not to copy the course into this repo. It is to identify source-grounded
ideas that can make `notebooks/99_complete_college_level_walkthrough.ipynb` more
self-contained, explanatory, and useful as a textbook-style overview.

## Reviewed Source Surfaces

- `data/cs124/README.md`: top-level map from CS124 lectures, SLP3 chapters, labs, and
  PAs to this project. It marks tokenization, n-gram LMs, embeddings, neural networks,
  transformers, LLMs, and contextual embeddings as directly relevant to the existing
  notebook path.
- `data/cs124/slp3_chapters/2.pdf`: words, tokens, Unicode, BPE, regular expressions,
  and edit distance.
- `data/cs124/slp3_chapters/3.pdf`: n-gram language models, train/dev/test splits,
  data contamination, perplexity, sampling, smoothing, interpolation, and backoff.
- `data/cs124/slp3_chapters/6.pdf`: neural networks, embedding matrices, one-hot
  lookup view, pooling and concatenation over embeddings, and backpropagation.
- `data/cs124/slp3_chapters/7.pdf`: large language model framing, prompting,
  decoding, temperature sampling, pretraining, SFT, alignment, and evaluation.
- `data/cs124/slp3_chapters/10.pdf`: contextual embeddings, word-instance
  representations, cosine similarity issues, and anisotropy.
- `data/cs124/slp3_chapters/12.pdf`: encoder-decoder transformers, cross-attention,
  teacher forcing, and generation/search.
- `data/cs124/lectures/tokens_jan26.pdf`, `lm_jan25.pdf`, `vector26jan.pdf`,
  `7_NN.pdf`, `transformer_cs124_26.pdf`, `llm_cs124_26.pdf`, and
  `ContextualEmbeddings25.pdf`: lecture-slide versions of the same core ideas.
- `data/cs124/pa6a-transformers/README.md`, `model.py`, and `perplexity.py`: runnable
  assignment scaffolding for causal attention, GPT-style training, sampling, and
  perplexity.
- `data/cs124/pa7-agent/README.md` and `agent.py`: tool-use agent, web search, memory,
  nondeterminism, and trajectory inspection.
- `data/cs124/labs/Lab2_LogisticRegression.md` and `Lab3_InformationRetrieval.md`:
  logit-to-probability intuition, threshold decisions, tf-idf versus dense retrieval,
  and precision/recall tradeoffs.

## Integration Principles

1. Prefer paraphrase with local source citations over copied prose.
2. Integrate ideas only where they improve the undergraduate `99` notebook.
3. Keep notebook examples CPU-friendly and avoid adding expensive dependencies.
4. Preserve the 13-section journey of notebook `99`; add depth inside that arc.
5. Avoid turning notebook `99` into a CS124 replica. The notebook remains an
   `llm-from-scratch` textbook companion.

## Extracted Ideas And Integration Text

### 1. Tokenization Needs The Type/Instance Distinction

**Source grounding**

- SLP3 Ch.2 introduces word types as vocabulary entries and word instances as
  occurrences in running text; it also notes that older terminology sometimes used
  "word token" for instances, while modern subword systems reserve "token" for
  algorithmic output units. See `slp3_chapters/2.pdf`, pages 2-3.
- The CS124 tokenization lecture emphasizes that larger corpora keep revealing more
  word types, which creates an unavoidable unknown-word problem for fixed word
  vocabularies. See `lectures/tokens_jan26.pdf`, pages 10 and 14.
- The same lecture frames BPE and unigram tokenization as two-stage systems: a learner
  induces a vocabulary from raw training text, and an encoder segments new text using
  that vocabulary. See `lectures/tokens_jan26.pdf`, page 46.
- The BPE/Unicode discussion says byte-level BPE avoids unknown tokens because there
  are only 256 byte values, while learned merges recover common multi-byte and
  multi-character units. See `lectures/tokens_jan26.pdf`, page 56.

**Notebook integration**

Notebook `99` should distinguish:

- a type: an entry in the vocabulary;
- an instance: an occurrence in a corpus;
- a token ID: the integer emitted by the tokenizer;
- a tokenization scheme: the rule or learned algorithm that maps strings to IDs.

This improves the tokenization section because it makes clear that `V` is not a
property of English, Python strings, or the model architecture. It is a property of
the chosen tokenizer and training corpus.

### 2. N-Gram Models Are The Right Baseline Before Neural LMs

**Source grounding**

- SLP3 Ch.3 defines a language model as a model that assigns probability to word or
  token sequences and predicts upcoming words. See `slp3_chapters/3.pdf`, pages 1-2.
- The chapter derives the chain-rule factorization and then motivates the n-gram
  approximation as replacing the full history with a bounded recent history. See
  `slp3_chapters/3.pdf`, pages 3-5.
- The lecture calls n-grams a simple model that introduces train/test sets,
  perplexity, sampling, interpolation, and backoff for later LLMs. See
  `lectures/lm_jan25.pdf`, page 19.
- SLP3 Ch.3 states that training on the test set, also called data contamination,
  biases probability-based evaluation and makes perplexity misleading. See
  `slp3_chapters/3.pdf`, page 8.
- The same chapter defines perplexity as a length-normalized inverse probability and
  explains that lower perplexity corresponds to higher probability assigned to the
  held-out text. See `slp3_chapters/3.pdf`, pages 9-10.

**Notebook integration**

Notebook `99` already begins with the exact autoregressive chain rule. It should now
add the count-model baseline:

```text
Exact chain rule:
P(x_1:T) = product_t P(x_t | x_<t)

N-gram approximation:
P(x_t | x_<t) approx P(x_t | x_{t-n+1:t-1})

Neural LM:
P_theta(x_t | x_<t) is produced by a learned network over a finite context window.
```

This makes the neural model easier to understand: the transformer is not the first
language model. It is a learned conditional probability estimator that replaces sparse
count tables with embeddings, attention, nonlinear layers, and gradient descent.

### 3. Perplexity Should Be Tied To Both Cross-Entropy And Tokenization

**Source grounding**

- PA6a asks students to compute perplexity from logits by log-softmaxing over the
  vocabulary, selecting the true next-token log-probabilities, averaging negative log
  probability, and exponentiating. See `pa6a-transformers/perplexity.py`, lines 54-91.
- PA6a also warns that perplexity and burstiness are weak detectors of AI writing
  because using predictability as a detector embeds assumptions about writing style.
  See `pa6a-transformers/README.md`, lines 100-120.
- CS124's LLM lecture notes that perplexity is sensitive to length and tokenization.
  See `lectures/llm_cs124_26.pdf`, page 86.
- The tokenization lecture says algorithms like perplexity assume fixed tokenization.
  See `lectures/tokens_jan26.pdf`, page 45.

**Notebook integration**

The notebook should state that when cross-entropy is averaged per token in natural-log
units,

```text
perplexity = exp(mean negative log-likelihood).
```

But it should also warn that perplexity comparisons are meaningful only when the
tokenization, test distribution, and evaluation protocol are comparable.

### 4. Embeddings Need Static Versus Contextual Framing

**Source grounding**

- SLP3 Ch.6 describes an embedding matrix `E` with one row per vocabulary item and
  shape `[|V| x d]`; one-hot multiplication selects the corresponding row. See
  `slp3_chapters/6.pdf`, pages 13-14.
- The vector-semantics lecture emphasizes cosine as a normalized dot product that
  avoids raw dot product's bias toward long vectors. See `lectures/vector26jan.pdf`,
  pages 46-50.
- The contextual-embeddings lecture contrasts static embeddings, which represent word
  types, with contextual embeddings, which represent word instances. See
  `lectures/ContextualEmbeddings25.pdf`, pages 3-10.
- SLP3 Ch.10 repeats that contextual embeddings are per-token-in-context
  representations, not one fixed dictionary vector per word type. See
  `slp3_chapters/10.pdf`, page 9.

**Notebook integration**

Notebook `99` currently explains embedding lookup. It should also say:

- the input embedding table is initially a type-level lookup table;
- after transformer layers, the residual stream contains contextualized token-instance
  vectors;
- similarity in embedding space is useful but not automatically semantic truth;
- cosine is a normalized geometric comparison, not a probability.

### 5. Logistic Regression Is The Small Bridge From Logits To Decisions

**Source grounding**

- CS124 Lab 2 defines a real-valued score `z = w dot x + b`, maps it to a probability
  through the sigmoid, and calls `z` the logit. See `labs/Lab2_LogisticRegression.md`,
  lines 9-32.
- The same lab emphasizes that probability is not a decision: a classifier still
  needs a threshold, and threshold choice can encode fairness and operational
  tradeoffs. See `labs/Lab2_LogisticRegression.md`, lines 258-278.
- The neural-network lecture uses multinomial logistic regression as a one-layer
  network with a softmax output. See `lectures/7_NN.pdf`, pages 38-41.

**Notebook integration**

The softmax/cross-entropy section should briefly identify logistic regression as the
single-output sibling of the vocabulary softmax:

- binary logit -> sigmoid -> probability of class 1;
- vocabulary logit vector -> softmax -> probability distribution over tokens;
- probability distribution -> loss/evaluation, not automatically a final decision.

### 6. Attention Is Information Movement Through A Residual Stream

**Source grounding**

- The transformer lecture defines query, key, and value as three projected roles of
  the same token vector: the current position asks a query, previous positions supply
  keys and values. See `lectures/transformer_cs124_26.pdf`, pages 16-19.
- The same lecture frames the residual stream as the vector carried upward for each
  token and says attention is the block component that moves information between
  token positions. See `lectures/transformer_cs124_26.pdf`, pages 29-35.
- PA6a's `CausalSelfAttention` scaffold requires students to project Q/K/V in batch,
  reshape to `(B, n_head, T, head_size)`, build scores of shape `(B, n_head, T, T)`,
  apply a causal mask, softmax over sources, multiply by values, and reshape to
  `(B, T, C)`. See `pa6a-transformers/model.py`, lines 29-82.
- The transformer lecture notes that attention is quadratic in sequence length. See
  `lectures/transformer_cs124_26.pdf`, page 40.

**Notebook integration**

The attention section should make the residual-stream view explicit:

```text
Layer-local MLPs transform each token vector independently.
Attention is the operation that mixes information across token positions.
The causal mask controls which source positions are allowed to write information into
the receiving position.
```

This is a strong conceptual bridge for undergraduates because it explains why attention
is not "just another matrix multiply": it is the only operation in the decoder block
that changes a token representation using other tokens.

### 7. Generation Should Separate Greedy Decoding, Sampling, And Temperature

**Source grounding**

- SLP3 Ch.7 describes prompts as conditioning text passed to a language model, after
  which the model iteratively generates tokens conditioned on prompt plus generated
  prefix. See `slp3_chapters/7.pdf`, pages 6-8.
- The chapter explains greedy decoding as choosing the highest-probability next token
  and notes that greedy decoding is deterministic and can become repetitive. See
  `slp3_chapters/7.pdf`, page 11.
- It contrasts random multinomial sampling with greedy decoding and motivates
  temperature sampling as logit scaling before softmax. See `slp3_chapters/7.pdf`,
  pages 11-14.
- PA6a's GPT implementation applies temperature by dividing the final-position logits
  before softmax, optionally applies top-k truncation, then samples with
  `torch.multinomial`. See `pa6a-transformers/model.py`, lines 313-337.

**Notebook integration**

The generation section should say:

- training optimizes likelihood; generation is a separate decoding policy;
- greedy decoding exposes the model's argmax behavior but removes stochasticity;
- random sampling respects the distribution but can sample bad low-probability tails;
- temperature changes entropy by scaling logits before softmax;
- top-k/top-p are distribution truncation policies, not training objectives.

### 8. Modern LLM Practice Needs A Tool-Use And Retrieval Boundary

**Source grounding**

- PA7 defines an LLM agent assignment where the model reasons about a task, chooses
  relevant tools, calls them, observes results, and produces a response. See
  `pa7-agent/README.md`, lines 165-236.
- PA7 adds web search because the base model may lack current external information,
  and it separates search, page parsing, and answer generation. See
  `pa7-agent/README.md`, lines 296-350.
- PA7 also introduces memory because a stateless agent cannot remember past
  interactions without an external memory mechanism. See `pa7-agent/README.md`,
  lines 352-365.
- Lab 3 contrasts exact lexical tf-idf retrieval with dense retrieval, where vector
  similarity can connect semantically related text that does not share exact surface
  words. See `labs/Lab3_InformationRetrieval.md`, lines 48-64.

**Notebook integration**

Notebook `99` should not implement an agent. It should add a conceptual boundary:

```text
The language model maps context tokens to next-token probabilities.
A retrieval system maps a query to external text.
A tool-use policy maps an intent to external function calls.
An agent system coordinates all three, plus state, permissions, and evaluation.
```

This keeps "LLM" from becoming an overloaded term for every surrounding product
system.

## Selected Integration Targets For Notebook 99

1. Add a classical n-gram bridge after the initial autoregressive chain-rule section.
2. Deepen tokenization with type/instance/token-ID/BPE/unknown-word distinctions.
3. Add a static-versus-contextual embedding bridge after embedding lookup.
4. Add a residual-stream interpretation after attention and transformer block sections.
5. Strengthen generation with decoding policy distinctions.
6. Add a compact retrieval/tool-use boundary in the modern LLM practice section.
7. Add a CS124-backed study checklist at the end that points to local source surfaces.

## Deferred Material

- Speech processing, collaborative filtering, and social networks are valuable but not
  central to notebook `99`'s undergrad LLM-from-scratch arc.
- Full encoder-decoder machine translation belongs in a later notebook or appendix.
- RAG should be introduced conceptually here, but implementation belongs elsewhere.
- Agent memory and web search should stay conceptual in `99`; PA7 can inform future
  exercises.
