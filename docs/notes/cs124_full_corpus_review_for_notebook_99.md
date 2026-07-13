# CS124 Full Corpus Review For Notebook 99

Date: 2026-06-22

Target notebook: `notebooks/99_complete_college_level_walkthrough.ipynb`

Source corpus: `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124`

This note records a broader pass over the local CS124 corpus after the first selective
undergraduate enrichment pass in `docs/notes/cs124_99_undergrad_enrichment.md`.

The goal is not to absorb all of CS124 into notebook 99. The goal is to extract
source-grounded context that makes notebook 99 more explanatory, self-contained, and
honest about the surrounding NLP landscape.

## Corpus Reconciliation

The top-level `README.md` and `catalog.json` say the downloaded corpus contains 72
PDFs. The live filesystem contains 74 PDFs:

| Bucket | PDFs | Pages | Extracted text chars | Status |
| --- | ---: | ---: | ---: | --- |
| `lectures/` | 21 | 1609 | 1354069 | course lecture slides |
| `slp3_chapters/` | 36 | 1424 | 6429252 | individual SLP3 chapters, appendices, and full book |
| `slp3_slides/` | 13 | 978 | 949412 | SLP3 teaching slides |
| `external/` | 2 | 70 | 116284 | web/recommendation readings |
| `labs/` PDF handouts | 2 | 40 | 13410 | extra local lab PDFs |
| **Total** | **74** | **4121** | **8862427** | all text-extractable |

Authority caveat: the README is a useful map, but it is stale in at least two ways.
First, it reports 72 PDFs while the filesystem has 74 because two lab PDFs are present.
Second, its SLP3 chapter map does not match the January 2026 chapter PDFs. For this
review, the current filesystem and extracted PDF title pages are treated as authority.

The full-book PDF `slp3_chapters/ed3book_jan26.pdf` duplicates the individual chapter
PDFs. Use individual chapters as the citation surface for notebook 99, and keep the
full book only as a convenience.

## Source Families

### Directly Useful For Notebook 99

These sources strengthen the notebook's core arc from text to tokens to LMs to
transformers to modern systems:

| Source | What it contributes |
| --- | --- |
| `slp3_chapters/2.pdf` and `lectures/tokens_jan26.pdf` | Unicode, regex, type/instance distinctions, edit distance, BPE, unknown-word handling |
| `slp3_chapters/3.pdf`, `slp3_chapters/C.pdf`, `lectures/lm_jan25.pdf` | n-gram LMs, train/dev/test protocol, perplexity, smoothing, interpolation, backoff, Kneser-Ney |
| `slp3_chapters/4.pdf`, `slp3_chapters/B.pdf`, `lectures/logreg26jan.pdf`, `labs/Lab2_LogisticRegression.md` | classification logits, sigmoid/softmax, threshold decisions, naive Bayes/logistic regression baselines |
| `slp3_chapters/5.pdf`, `slp3_chapters/6.pdf`, `lectures/vector26jan.pdf`, `slp3_slides/vector25aug.pdf` | embedding matrices, cosine similarity, distributional semantics, neural network bridge |
| `slp3_chapters/7.pdf`, `slp3_chapters/8.pdf`, `lectures/llm_cs124_26.pdf`, `lectures/transformer_cs124_26.pdf`, `slp3_slides/transformer_jan26.pdf` | LLM framing, decoder-only transformers, residual stream, Q/K/V, attention cost, prompting, decoding |
| `slp3_chapters/9.pdf`, `lectures/finallecture_cs124_2025.pdf`, `lectures/final124_26.pdf` | post-training, instruction tuning, alignment, preference methods, test-time compute, recent LLM practice |
| `slp3_chapters/10.pdf`, `slp3_slides/mlmjan25.pdf`, `lectures/ContextualEmbeddings25.pdf` | masked language modeling, BERT, bidirectional encoders, contextual representations |
| `slp3_chapters/11.pdf`, `slp3_slides/ir_nov25.pdf`, `lectures/ir_jan17_2026.pdf`, `labs/Lab3_InformationRetrieval.md` | inverted indexes, tf-idf/BM25, dense retrieval, retrieval-augmented generation |
| `slp3_chapters/12.pdf` | encoder-decoder MT, teacher forcing, beam search, cross-attention context |
| `pa1-regular-expressions/`, `pa3-information-retrieval/`, `pa6a-transformers/`, `pa7-agent/` | runnable assignment contracts for BPE, IR, attention/perplexity/generation, tools/web search/memory |

### Useful As Boundary Context, Not Core Notebook Material

These sources explain what an LLM-from-scratch notebook leaves out:

| Source | Boundary lesson for notebook 99 |
| --- | --- |
| `slp3_chapters/13.pdf` | RNNs/LSTMs are sequential baselines and historical context for transformers |
| `slp3_chapters/14.pdf`, `15.pdf`, `16.pdf`, `lectures/speech_cs124_26.pdf`, `pa6b-speech/` | speech has signal-processing and accessibility concerns beyond text token modeling |
| `slp3_chapters/17.pdf` | POS/NER are structured prediction tasks over token sequences |
| `slp3_chapters/18.pdf`, `19.pdf`, `E.pdf`, `F.pdf`, `G.pdf` | grammar and parsing model explicit syntax rather than only next-token likelihood |
| `slp3_chapters/20.pdf`, `21.pdf`, `H.pdf`, `I.pdf` | information extraction, semantic roles, logical form, word senses add explicit semantic structure |
| `slp3_chapters/22.pdf`, `23.pdf`, `24.pdf`, `25.pdf`, `K.pdf` | sentiment, coreference, entity linking, discourse, conversation, and task dialogue define application-level behaviors |
| `lectures/collaborativefiltering26.pdf`, `lectures/socialnetworks21.pdf`, `external/ch9.pdf`, `labs/GW5_Chatbots.md`, `labs/Lab5_LargeLanguageModels.md` | recommendation, social networks, and AI policy are adjacent systems topics, not core LM mechanics |

## Extracted Concepts For Notebook 99

### 1. Classical String Processing Still Matters

Source grounding:

- `slp3_chapters/2.pdf` introduces Unicode, UTF-8, regular expressions, edit distance,
  and BPE in the same opening text-processing chapter.
- `lectures/med25.pdf` and `slp3_chapters/D.pdf` add minimum edit distance and noisy
  channel spelling correction.
- `pa1-regular-expressions/pa1.ipynb` and `regular_expressions_tutorial.ipynb` turn
  the tokenization material into executable exercises.

Notebook consequence:

Notebook 99 should not imply that token IDs are the first serious NLP abstraction.
Before neural modeling, the pipeline must still decide how strings are normalized,
matched, segmented, and compared. Regex, Unicode, edit distance, and BPE are all ways
to impose structure on raw strings before a transformer sees integer IDs.

Recommended integration:

Add a short section after tokenization explaining:

- deterministic string tools operate before learned token IDs;
- BPE is statistical string compression, not semantic understanding;
- edit distance and noisy-channel correction are classical examples of modeling
  spelling variation without embeddings;
- bad string assumptions leak into every later tensor.

### 2. Smoothing Is The Count-Model Version Of Inductive Bias

Source grounding:

- `slp3_chapters/3.pdf` introduces smoothing, interpolation, and backoff after showing
  the sparse maximum-likelihood n-gram estimate.
- `slp3_chapters/C.pdf` gives Kneser-Ney as an advanced n-gram smoothing method rooted
  in discounting and continuation behavior.
- `lectures/lm_jan25.pdf` uses n-grams to introduce train/test splits, perplexity,
  sampling, interpolation, and backoff.

Notebook consequence:

The current n-gram bridge explains the count estimate. It should also explain why the
count estimate fails: unseen events receive zero probability. Smoothing is the
classical answer to finite data. A transformer's learned parameters are more flexible,
but the problem is the same: assign reasonable probability to held-out strings that
were not memorized exactly.

Recommended integration:

Add a short section after the n-gram bridge:

```text
Maximum-likelihood n-gram:
P(w | h) = count(h,w) / count(h)

Failure:
if count(h,w) = 0, the model assigns zero probability.

Smoothing:
move probability mass away from seen events so unseen or rare events can receive
nonzero probability.
```

Tie this to neural LMs as an inductive-bias problem, not just an old formula.

### 3. Architecture Families Need A Clear Map

Source grounding:

- `slp3_chapters/8.pdf` covers transformers and explicitly frames decoder-only
  transformer language modeling.
- `slp3_chapters/10.pdf` covers masked language models such as BERT as bidirectional
  transformer encoders trained with a mask-prediction objective.
- `slp3_chapters/12.pdf` covers machine translation through encoder-decoder models,
  teacher forcing, and beam search.
- `slp3_slides/transformer_jan26.pdf`, `slp3_slides/mlmjan25.pdf`, and
  `slp3_slides/rnnjan25.pdf` provide slide-level contrasts.

Notebook consequence:

Notebook 99 is centered on decoder-only GPT-style modeling. That is correct for the
project, but the reader should know it is one member of a transformer family:

| Family | Attention pattern | Common objective | Example use |
| --- | --- | --- | --- |
| Encoder-only | bidirectional self-attention | masked-token or representation learning | classification, retrieval encoding |
| Decoder-only | causal self-attention | next-token prediction | text generation, chat base models |
| Encoder-decoder | encoder self-attention plus decoder causal/cross-attention | conditional generation | translation, summarization |

Recommended integration:

Add a compact section near the Hugging Face translation or transformer abstraction
section so the notebook's GPT focus is explicit rather than accidental.

### 4. Retrieval Should Be More Concrete Than "External Context"

Source grounding:

- `slp3_chapters/11.pdf` covers information retrieval and retrieval-augmented
  generation, including sparse count-vector retrieval, BM25/tf-idf weighting, dense
  retrieval, inverted indexes, and RAG's retrieve-then-generate decomposition.
- `slp3_slides/ir_nov25.pdf` and `lectures/ir_jan17_2026.pdf` provide the course-slide
  version.
- `labs/Lab3_InformationRetrieval.md` makes students compare tf-idf cosine retrieval
  against dense retrieval and reason about precision/recall.
- `pa3-information-retrieval/` contains the assignment scaffold and query files.

Notebook consequence:

The existing retrieval/tool boundary is correct, but it can be more self-contained by
showing the retrieval contract:

```text
Sparse retrieval:
query/document strings -> weighted term vectors -> nearest documents by lexical match.

Dense retrieval:
query/document strings -> learned vectors -> nearest documents by embedding similarity.

RAG:
question -> retriever -> passages -> LM context -> answer.
```

This also clarifies that a RAG failure can come from missing documents, bad chunking,
ranking errors, context-window packing, or generation, not just the model.

### 5. Post-Training Is A Separate Learning Stage

Source grounding:

- `slp3_chapters/9.pdf` is explicitly about post-training: instruction tuning,
  alignment, and test-time compute.
- `lectures/finallecture_cs124_2025.pdf` and `lectures/final124_26.pdf` cover recent
  LLM practice beyond pretraining.
- `labs/Lab5_LargeLanguageModels.md` covers policy and classroom-use stress tests,
  which are not model mechanics but do matter for deployment.

Notebook consequence:

Notebook 99 already covers SFT/preference optimization conceptually. It should name
the stage boundary more clearly:

- pretraining learns broad next-token continuation behavior;
- instruction tuning changes the data distribution toward instruction-response
  behavior;
- preference/alignment methods optimize over comparative output quality or constraints;
- test-time compute changes inference behavior without changing weights.

Recommended integration:

Add a short CS124-backed post-training paragraph/table in the modern LLM practice
section, after the preference optimization discussion.

### 6. Agents Are System Loops Around Models

Source grounding:

- `pa7-agent/README.md` asks students to build a tool-use movie-ticket agent with
  tool trajectories, database interaction, web search, memory, and nondeterminism.
- `labs/Lab4_Agent.md` introduces a trajectory with thoughts, tool names, tool
  arguments, observations, and finish steps.
- `pa7-agent/agent.py` includes separate tool, web-search, and memory utilities.
- `slp3_chapters/25.pdf` and `slp3_chapters/K.pdf` provide dialogue/conversation
  context, including turns, task dialogue, frames, slots, and dialogue state.

Notebook consequence:

An agent is not just a larger decoder. It is a loop that lets a model condition on
observations produced by external actions. Notebook 99 should preserve this boundary:

```text
LM forward pass: context tokens -> next-token distribution.
Agent loop: state -> model call -> tool decision -> external action -> observation -> state update.
```

The PA7 materials make the engineering risks concrete: tools have schemas, side
effects, permissions, failures, and nondeterministic model calls.

### 7. The Rest Of NLP Provides Evaluation Pressure

Source grounding:

- `slp3_chapters/17.pdf` through `25.pdf` cover sequence labeling, parsing,
  information extraction, semantic roles, sentiment/connotation, coreference/entity
  linking, discourse coherence, and conversation.
- `slp3_chapters/14.pdf` through `16.pdf` cover speech feature extraction,
  recognition, and text-to-speech.
- `lectures/speech_cs124_26.pdf`, `pa6b-speech/`, and `labs/GW5_Chatbots.md` add
  application context outside pure text generation.

Notebook consequence:

Notebook 99 should include a "what this leaves out" table. This makes the textbook
more honest: next-token prediction is a powerful objective, but NLP also contains
tasks where structure, grounding, interaction, and evaluation are explicit.

Recommended table:

| Area | What it asks | Why it matters to LLMs |
| --- | --- | --- |
| Sequence labeling | assign labels to token positions | many "understanding" tasks are structured outputs |
| Parsing | recover syntactic structure | long-range grammatical relations are not just adjacent token statistics |
| Information extraction | extract entities, relations, events, and times | useful outputs often need schemas |
| Semantics | represent roles, senses, and meanings | generation quality is not the same as semantic correctness |
| Coreference/discourse | track entities and coherence across text | long answers need stable references and structure |
| Speech | map between waveforms and text | language systems often start or end outside text |
| Dialogue/agents | coordinate turns, tools, memory, and goals | applications are loops, not one forward pass |

## Integration Decisions For This Pass

Added to notebook 99:

1. A string-processing bridge after tokenization.
2. A smoothing/data-sparsity bridge after the n-gram baseline.
3. A transformer-family map distinguishing encoder-only, decoder-only, and
   encoder-decoder models.
4. A concrete sparse/dense/RAG retrieval bridge.
5. A post-training stage table in modern LLM practice.
6. A broad "NLP around the core LM" table near the end.

Not added:

- Full speech processing: valuable, but it would require signal-processing material
  that is outside notebook 99's text-to-transformer path.
- Full parsing and formal semantics derivations: useful later, but too large for the
  current undergrad overview.
- Collaborative filtering and social networks: adjacent to recommendation/agent labs,
  not central to the LM mechanics.
- External web/recommendation PDFs: preserved as adjacent context, not integrated.
- Full RAG implementation: notebook 99 should name the contract; implementation belongs
  in a separate retrieval notebook or exercise.

## Restart Notes

If continuing this work, inspect these files first:

1. `notebooks/99_complete_college_level_walkthrough.ipynb`
2. `docs/notes/cs124_99_undergrad_enrichment.md`
3. `docs/notes/cs124_full_corpus_review_for_notebook_99.md`
4. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/2.pdf`
5. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/3.pdf`
6. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/8.pdf`
7. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/9.pdf`
8. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/10.pdf`
9. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/11.pdf`
10. `/Users/tovarishchrafa/Desktop/human-learning-machine-learning/data/cs124/slp3_chapters/12.pdf`

The smallest useful next pass would be a dedicated retrieval/RAG notebook seeded from
`slp3_chapters/11.pdf`, `lectures/ir_jan17_2026.pdf`, `labs/Lab3_InformationRetrieval.md`,
and `pa3-information-retrieval/`.
