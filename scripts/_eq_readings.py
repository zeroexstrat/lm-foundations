# ═══════════════════════════════════════════════════════════════════════
#  EQUATION READINGS — plain-English "how to read it" for every formula
#  All spacing verified — no joined words, proper punctuation spacing.
# ═══════════════════════════════════════════════════════════════════════

EQ_READINGS: dict[str, str] = {
    # ── Chapter 0: What A Language Model Computes ──
    r"x_{1:T} = (x_1, x_2, \ldots, x_T), \qquad x_t \in \{0,1,\ldots,V-1\},": (
        "A token sequence of length T. The symbol x sub t (each position's token) "
        "is an integer between 0 and V−1, where V is the vocabulary size. "
        "The notation x sub 1 colon T means the whole sequence from position 1 through T."
    ),
    r"p_\theta(x_t \mid x_{<t}),": (
        "The probability that the next token is x sub t, given all previous tokens x sub less-than-t, "
        "with all model parameters collectively denoted by theta. "
        "The vertical bar means 'conditioned on': the model looks at the prefix and predicts what comes next."
    ),
    r"\mathcal{V} = \{0,1,\ldots,V-1\}.": (
        "The vocabulary V is the set of all possible token IDs: the integers 0 through V−1. "
        "Calligraphic V denotes the set itself; V denotes its size."
    ),
    r"P_\theta(X_{1:T}=x_{1:T}) = \prod_{t=1}^{T} P_\theta(X_t=x_t \mid X_{<t}=x_{<t}).": (
        "The chain rule of probability: the probability of the entire sequence equals the product "
        "of each token's conditional probability given all previous tokens. "
        "Capital P means probability; the product symbol (big pi) means multiply these T terms together."
    ),
    r"\log P_\theta(x_{1:T}) = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}).": (
        "Taking the logarithm turns the product into a sum. The log of a product equals the sum of the logs. "
        "This sum of per-token log-probabilities is what the model optimizes — each token contributes one term."
    ),
    r"-\log P_\theta(x_{1:T}) = \sum_{t=1}^{T} -\log P_\theta(x_t \mid x_{<t}).": (
        "Negative log-likelihood: we negate so that maximizing probability becomes minimizing loss. "
        "This is the per-token cross-entropy objective — the core training signal for language models."
    ),
    r"\sum_{i\in A(x)} \log p_\theta(x_i\mid \bar{x}),": (
        "Masked language modeling objective: sum the log-probabilities only for a selected subset A(x) of masked positions. "
        "The model sees the corrupted input x-bar and predicts only the masked tokens. "
        "This is how BERT-style encoders are trained, in contrast to causal (autoregressive) prediction."
    ),
    r"\texttt{idx}[b,t] \in \{0,1,\ldots,V-1\}.": (
        "A batch of token IDs: idx at batch index b and position t is an integer token ID. "
        "The batch dimension B lets the model process multiple sequences in parallel."
    ),
    r"\Delta^{V-1} = \left\{p \in \mathbb{R}^{V}: p_i \ge 0,\; \sum_{i=0}^{V-1}p_i=1\right\}.": (
        "The (V−1)-dimensional probability simplex. "
        "Every vector p of length V whose entries are non-negative and sum to 1 lives in this set. "
        "The model's output after softmax is always a point in this simplex — a valid probability distribution."
    ),
    r"\widehat{R}(\theta) = \frac{1}{N}\sum_{n=1}^{N}\frac{1}{T_n-1}\sum_{t=1}^{T_n-1} -\log p_\theta(x^{(n)}_{t+1}\mid x^{(n)": (
        "Empirical risk: the average negative log-likelihood over N training documents. "
        "For document n of length T_n, we average over all T_n−1 next-token predictions. "
        "The hat on R means this is estimated from data, not the true population risk."
    ),
    r"\widehat{R}_{\text{batch}}(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta(y_{b,t}\mid x_{b,1:t}).": (
        "Batch empirical risk: same idea but computed over a mini-batch of B sequences each of length T. "
        "This is exactly the scalar value that autograd differentiates to produce parameter gradients."
    ),

    # ── Chapter 0: N-Grams ──
    r"P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}).": (
        "The exact chain rule again. Every autoregressive model — from n-grams to GPT-4 — "
        "factors sequence probability this way. What differs is how each conditional factor is approximated."
    ),
    r"P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-n+1:t-1}).": (
        "The n-gram Markov assumption: the next token depends only on the previous n−1 tokens, "
        "not the entire history. This truncates the context window to a fixed length."
    ),
    r"\hat P(w_t \mid h) = \frac{\operatorname{count}(h, w_t)}{\operatorname{count}(h)},": (
        "Maximum likelihood estimate for an n-gram: count how many times the history h is followed by word w_t, "
        "divided by how many times history h appears at all. A simple ratio of corpus statistics."
    ),
    r"\hat P(w \mid h)=\frac{\operatorname{count}(h,w)}{\operatorname{count}(h)}.": (
        "Same n-gram estimate, compact notation. count(h,w) tallies the (history, word) bigram occurrences; "
        "count(h) tallies the history alone. When count(h) = 0, we need smoothing."
    ),

    # ── Chapter 1: Tensors, Autograd, Probability ──
    r"L(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \ell\left(z_\theta(b,t), y_{b,t}\right),": (
        "The scalar training loss: average over B batches and T positions of a per-example loss function ℓ. "
        "z_θ(b,t) is the logit vector at position (b,t); y_{b,t} is the correct token ID. "
        "This scalar is what backward() differentiates."
    ),
    r"\ell(z,y) = -\log \operatorname{softmax}(z)_y.": (
        "Cross-entropy loss: take the softmax of logits z to get probabilities, "
        "extract the probability of the correct class y, and take the negative log. "
        "Lower loss means the model assigned higher probability to the correct answer."
    ),
    r"\theta \mapsto z_\theta \mapsto L(\theta).": (
        "The computation graph: parameters θ produce logits z_θ, which produce a scalar loss L. "
        "Backpropagation reverses these arrows, flowing gradients from L back to θ."
    ),
    r"\theta_{k+1} = \theta_k - \eta \nabla_\theta L(\theta_k),": (
        "Gradient descent update rule: parameters at step k+1 equal parameters at step k "
        "minus the learning rate η times the gradient of the loss. "
        "We step downhill in the direction that most reduces the loss."
    ),

    # ── Softmax ──
    r"\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}.": (
        "Softmax: for each vocabulary item i, exponentiate its logit z_i and divide by the sum of "
        "exponentiated logits over all V items. This guarantees outputs are positive and sum to 1 — a valid probability distribution."
    ),
    r"\operatorname{softmax}(z + c\mathbf{1})_i = \frac{e^{z_i+c}}{\sum_j e^{z_j+c}} = \frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}} = ": (
        "Proof that softmax is shift-invariant: adding the same constant c to every logit doesn't change the output. "
        "The e^c factor appears in both numerator and denominator and cancels out. "
        "This means we can always subtract the maximum logit for numerical stability without affecting the result."
    ),
    r"\operatorname{softmax}(z)_i = \frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)}, \qquad m = \max_j z_j.": (
        "The numerically stable softmax: subtract the maximum logit m from every entry before exponentiating. "
        "This prevents overflow because the largest exponentiated value is e⁰ = 1. "
        "The result is mathematically identical to standard softmax but safe to compute. "
        "Read as: softmax of z at position i equals exp of (z_i minus m), divided by the sum over j of exp of (z_j minus m)."
    ),
    r"\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j-m}, \qquad m=\max_j z_j.": (
        "Log-sum-exp trick: to compute log(Σ e^{z_j}) safely, subtract the maximum m from all z_j, "
        "compute the log-sum-exp of the shifted values, then add m back. "
        "This prevents overflow when some z_j are very large. Used in the numerically stable cross-entropy. "
        "Read as: log of the sum over j of e to the z_j equals m plus log of the sum over j of e to the (z_j minus m)."
    ),
    r"\frac{\partial p_i}{\partial z_k} = p_i(\mathbf{1}[i=k] - p_k).": (
        "The softmax Jacobian: how much probability p_i changes when logit z_k changes. "
        "If i = k (diagonal), the derivative is p_i(1−p_i). If i ≠ k (off-diagonal), it's −p_i·p_k. "
        "This elegant form makes backprop through softmax efficient."
    ),
    r"J = \operatorname{diag}(p) - pp^T.": (
        "The full softmax Jacobian matrix J: diagonal matrix of probabilities minus the outer product pp^T. "
        "This V×V matrix captures all pairwise dependencies between logits and probabilities."
    ),

    # ── Cross-Entropy ──
    r"L(z,y) = -\log p_y.": (
        "Cross-entropy loss in its simplest form: negative log of the model's probability for the correct class y. "
        "If p_y = 1 (perfect prediction), loss is 0. If p_y → 0, loss → ∞."
    ),
    r"L(z,y) = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = -z_y + \log\sum_j e^{z_j}.": (
        "Cross-entropy expanded: negative z_y (the correct class's logit) plus the log-sum-exp of all logits. "
        "This is the numerically stable form used in practice (with the max subtracted for stability)."
    ),
    r"\frac{\partial L}{\partial z_i} = -\mathbf{1}[i=y] + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - \mathbf{1}[i=y].": (
        "The celebrated gradient: ∂L/∂z_i = p_i − 1[i=y]. "
        "For the correct class (i = y), gradient is p_y − 1 (negative, pushing probability up). "
        "For wrong classes (i ≠ y), gradient is p_i (positive, pushing probability down). "
        "This clean form makes softmax + cross-entropy the workhorse of classification."
    ),
    r"H(q,p) = -\sum_i q_i \log p_i.": (
        "Cross-entropy between true distribution q and model distribution p: "
        "for each class i, multiply the true probability q_i by the log of the model's probability p_i, sum, and negate. "
        "When q is one-hot (classification), this reduces to −log p_y."
    ),
    r"H(q,p) = H(q) + \mathrm{KL}(q\|p),": (
        "Decomposition: cross-entropy = entropy of q + KL divergence from p to q. "
        "Since H(q) is constant (the data's entropy), minimizing cross-entropy is equivalent to minimizing KL divergence."
    ),
    r"\mathrm{KL}(q\|p) = \sum_i q_i \log\frac{q_i}{p_i} = \sum_i q_i\log q_i - \sum_i q_i\log p_i = -H(q) + H(q,p).": (
        "KL divergence: measures how much information is lost when using distribution p to approximate q. "
        "It's always non-negative and zero only when p = q. The asymmetry matters: KL(q‖p) ≠ KL(p‖q)."
    ),
    r"H(q,p) = -\log p_y.": (
        "Cross-entropy for one-hot targets: when q is a one-hot vector with q_y = 1, "
        "the sum collapses to a single term: −log p_y. This is the loss used in language model training."
    ),

    # ── Logistic Regression ──
    r"z = w\cdot x + b,": (
        "The linear predictor: logit z equals the dot product of weight vector w and input x, plus bias b. "
        "This is the simplest possible model — a single linear layer before the sigmoid."
    ),
    r"\sigma(z) = \frac{1}{1 + e^{-z}}.": (
        "The sigmoid (logistic) function: maps any real number z to the interval (0,1). "
        "When z is large positive, σ(z) ≈ 1. When z is large negative, σ(z) ≈ 0. "
        "This is the building block of logistic regression."
    ),

    # ── Chapter 2: Tokenization ──
    r"\operatorname{encode}: \text{strings} \to \mathcal{V}^T,": (
        "The encode function: maps a text string to a sequence of T token IDs from the vocabulary. "
        "This is the forward direction — raw text goes in, integer sequence comes out."
    ),
    r"\operatorname{decode}: \mathcal{V}^T \to \text{strings}.": (
        "The decode function: maps a sequence of token IDs back to a text string. "
        "Note this is not always perfectly invertible — some tokenizers (like BPE) may lose information."
    ),
    r"\text{scores per head} = T^2.": (
        "T² attention scores per head: each of the T query positions computes a score against each of the T key positions. "
        "This quadratic count is the main performance bottleneck in transformer models."
    ),

    # ── Chapter 3: Embeddings ──
    r"E: \mathcal{V} \to \mathbb{R}^C.": (
        "The embedding map: takes a token ID from the vocabulary and returns a C-dimensional vector. "
        "This turns discrete symbols into continuous coordinates where linear algebra applies."
    ),
    r"W_E \in \mathbb{R}^{V\times C}.": (
        "The embedding matrix: V rows (one per vocabulary item) × C columns (embedding dimension). "
        "Row i of this matrix is the learned vector representation of token i."
    ),
    r"E(i) = W_E[i,:].": (
        "Embedding lookup: the embedding of token i is simply the i-th row of the embedding matrix. "
        "In code, this is an efficient table lookup, but mathematically it's equivalent to one-hot multiplication."
    ),
    r"e_i W_E = W_E[i,:].": (
        "The one-hot embedding as matrix multiplication: multiplying the one-hot row vector e_i "
        "(a row of zeros with a single 1 at position i) by the embedding matrix W_E is equivalent to selecting row i. "
        "The neural network uses a fast table lookup, but mathematically it's a linear transformation."
    ),
    r"x_{b,t,:} = W_E[x_{b,t}],: + W_P[t,:].": (
        "The initial residual stream at batch b, position t: token embedding (row of W_E) "
        "plus position embedding (row of W_P). "
        "The colon means 'all C dimensions.' This combined vector enters the first transformer block."
    ),
    r"\text{same token ID} + \text{different context} \Rightarrow \text{different hidden vector}.": (
        "Context matters: the same word (like 'bank') at position 3 vs position 15 gets different hidden representations "
        "because attention incorporates different surrounding context. "
        "This is the core power of transformers over static embeddings."
    ),

    # ── Label Shifting ──
    r"(x_1,x_2,\ldots,x_{T+1})": (
        "A raw token sequence of length T+1. From this one sequence, we create T training examples "
        "by shifting: (x_1 → x_2), (x_1,x_2 → x_3), ..., (x_1,...,x_T → x_{T+1})."
    ),
    r"(x_1) \mapsto x_2,": (
        "Training example 1: given the first token x_1, predict x_2."
    ),
    r"(x_1,x_2) \mapsto x_3,": (
        "Training example 2: given tokens x_1 and x_2, predict x_3."
    ),
    r"(x_1,\ldots,x_T) \mapsto x_{T+1}.": (
        "Training example T: given all T prefix tokens, predict the final token x_{T+1}. "
        "This is label shifting: one text sequence yields many supervised prediction problems."
    ),
    r"L(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta(y_{b,t}\mid x_{b,\le t}).": (
        "The scalar training loss for one batch: average over B sequences and T positions "
        "of the negative log-probability of the correct token. This is exactly the value "
        "that backward() differentiates to produce parameter gradients. "
        "Read as: loss L of theta equals one over BT, sum from b=1 to B, sum from t=1 to T, "
        "of negative log p_theta of y at (b,t) given x from 1 to t in batch b."
    ),

    # ── Chapter 4: Attention ──
    r"Q,K,V \in \mathbb{R}^{T\times D}.": (
        "The Query, Key, and Value matrices: each is T×D (sequence length × per-head dimension). "
        "Queries ask 'what am I looking for?', Keys answer 'what do I contain?', "
        "Values carry the actual information that gets aggregated."
    ),
    r"a_{t,s} = q_t \cdot k_s.": (
        "The attention score between query position t and key position s: the dot product of their vectors. "
        "A high dot product means the query and key are well-aligned — position t finds position s relevant."
    ),
    r"A = QK^T \in \mathbb{R}^{T\times T}.": (
        "The full attention score matrix: T×T, where entry (t,s) is the dot product q_t·k_s. "
        "Computing this matrix costs O(T²·D) operations — this is the quadratic bottleneck of standard attention."
    ),
    r"W_{t,:} = \operatorname{softmax}\left(\frac{A_{t,:}}{\sqrt{D}} + \text{mask}_{t,:}\right).": (
        "Attention weights for query t: take the t-th row of scores A, divide by √D (scaling), "
        "add the mask (to block future positions), apply softmax to get a probability distribution over keys. "
        "Each row sums to 1 — a weighted average over value vectors."
    ),
    r"o_t = \sum_{s=1}^{T} W_{t,s}v_s.": (
        "The output at position t: a weighted sum of all value vectors, where the weights W_{t,s} "
        "come from the softmax-normalized attention scores. Positions with higher attention contribute more."
    ),
    r"O = WV.": (
        "The full output matrix: O = W·V, where W is the T×T attention weight matrix and V is the T×D value matrix. "
        "Each row of O is a context-dependent representation of one token position."
    ),
    r"w^* = \arg\max_{w\in\Delta^{T-1}} \left\{\sum_s w_s a_s + \tau H(w)\right\},": (
        "Entropy-regularized attention: choose weights w that maximize a trade-off between relevance (Σ w_s·a_s) "
        "and entropy H(w). The temperature τ controls how much we encourage diversity vs. focus. "
        "When τ → 0, this becomes hard attention (argmax). When τ = 1, this is exactly standard softmax attention."
    ),

    # ── Variance Scaling ──
    r"q\cdot k = \sum_{r=1}^{D} q_r k_r.": (
        "The dot product expanded: multiply corresponding entries of q and k, then sum all D products. "
        "For random vectors, this sum grows with D — which is precisely why we need the scaling factor."
    ),
    r"\operatorname{Var}(q\cdot k) \approx D.": (
        "Without scaling, the dot product's variance is roughly D (the dimension). "
        "When D is large (e.g., 64 or 128), this makes softmax inputs very large, saturating gradients."
    ),
    r"\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{D}}\right) \approx 1.": (
        "Dividing by √D brings the variance back to approximately 1. "
        "This keeps softmax inputs in a healthy range where gradients flow well. "
        "This is the key insight behind scaled dot-product attention."
    ),
    r"\frac{QK^T}{\sqrt{D}}.": (
        "Scaled attention scores: QK^T divided element-wise by √D. "
        "This single division is what makes deep transformers trainable — without it, gradients vanish."
    ),
    r"M_{t,s} = \begin{cases} 1, & s\le t,\\ 0, & s>t. \end{cases}": (
        "The causal mask M: 1 for positions s ≤ t (past and present), 0 for s > t (future). "
        "After converting 0 → −∞, softmax assigns exactly zero weight to future positions. "
        "This is what makes the model autoregressive — it cannot peek ahead."
    ),
    r"\operatorname{softmax}([\ldots, -\infty, \ldots]) = [\ldots, 0, \ldots].": (
        "Softmax of −∞ is exactly 0. The causal mask sets future positions' scores to −∞, "
        "so their attention weights become exactly 0. This enforces the constraint that "
        "position t cannot attend to any position t+1, t+2, ..., T."
    ),

    # ── Attention Bottleneck ──
    r"(B,H,T,T).": (
        "The attention weight tensor shape: Batch × Heads × Queries(T) × Keys(T). "
        "For each head in each batch, we have a T×T attention matrix stored in memory."
    ),
    r"BHT^2.": (
        "Attention memory cost: B batches × H heads × T² entries per attention matrix. "
        "This quadratic growth in T is why long sequences are so expensive in memory."
    ),
    r"2048^2 = 4,194,304": (
        "At sequence length 2048, each attention head computes over 4 million attention scores. "
        "With 32 heads, that's over 130 million scores per transformer layer. "
        "This is why longer contexts need GPU memory measured in tens of gigabytes."
    ),

    # ── Chapter 5: Transformer Block ──
    r"x \in \mathbb{R}^{B\times T\times C},": (
        "The residual stream: a tensor of shape Batch × Time × Channels (model dimension). "
        "Every transformer operation reads from and writes to this same-dimensional stream."
    ),
    r"x' = x + \operatorname{Attn}(\operatorname{LN}_1(x)),": (
        "The attention sublayer with pre-norm: normalize x (LayerNorm), compute attention, "
        "then add back to x (residual connection). "
        "The residual connection lets gradients flow directly through the block, avoiding vanishing gradients."
    ),
    r"x'' = x' + \operatorname{MLP}(\operatorname{LN}_2(x')).": (
        "The MLP sublayer with pre-norm: normalize the attention output, pass through the feedforward network, "
        "then add back via residual connection. This completes one full transformer block."
    ),
    r"\mu = \frac{1}{C}\sum_{i=1}^{C} u_i,": (
        "LayerNorm mean: average all C entries of the vector u. This centers the activations around zero."
    ),
    r"\sigma^2 = \frac{1}{C}\sum_{i=1}^{C}(u_i-\mu)^2,": (
        "LayerNorm variance: average squared deviation from the mean. This measures the spread of activations."
    ),
    r"\operatorname{LN}(u)_i = \gamma_i\frac{u_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i.": (
        "LayerNorm formula: for each dimension i, subtract the mean, divide by standard deviation (√(σ²+ε)), "
        "multiply by learned scale γ_i, add learned shift β_i. ε is a tiny constant (like 1e-5) preventing division by zero. "
        "The learnable γ and β let the network undo the normalization if it wants to."
    ),
    r"\operatorname{MLP}(u) = W_2\,\operatorname{GELU}(W_1u + b_1) + b_2.": (
        "The feedforward network: project input u from C to 4C dimensions (W_1), apply GELU activation, "
        "project back from 4C to C (W_2). GELU is a smooth approximation to ReLU that works better in practice."
    ),
    r"x \mapsto x + f(x).": (
        "A residual connection: the output is the input plus some function f of the input. "
        "If f learns to output zero, the block is an identity — this makes very deep networks trainable."
    ),
    r"x_{\text{out}} = \operatorname{LN}(x + F(x)).": (
        "Post-norm layout: apply the sublayer F first, add the residual, then normalize. "
        "This was the original Transformer design but is less stable for deep stacks than pre-norm."
    ),
    r"x_{\text{out}} = x + F(\operatorname{LN}(x)).": (
        "Pre-norm layout: normalize first, apply the sublayer F, then add the residual. "
        "This is the modern standard — gradients flow cleanly through the residual path to earlier layers."
    ),

    # ── Chapter 6: Training ──
    r"\theta_{k+1} = \theta_k - \eta \nabla L(\theta_k).": (
        "The gradient descent update: parameters at step k+1 equal parameters at step k, "
        "minus the learning rate η times the gradient of the loss at step k. "
        "We move downhill in the direction that most reduces the loss. "
        "This is the fundamental optimization step repeated thousands of times during training. "
        "Read as: theta sub k+1 equals theta sub k minus eta times the gradient of L at theta sub k."
    ),
    r"\operatorname{perplexity} = \exp(L).": (
        "Perplexity = exp(loss): exponentiate the average cross-entropy loss L. "
        "If perplexity = 10, the model is as uncertain as if it were choosing uniformly among 10 tokens. "
        "Lower perplexity means the model is more confident about its predictions."
    ),

    # ── Chapter 7: Generation ──
    r"p_\theta(x_{t+1}\mid x_{\le t}).": (
        "The core prediction: the probability of the next token x_{t+1} given all tokens up to position t. "
        "This single conditional probability is the building block — the model computes one of these for every position, "
        "and the chain rule turns them into a sequence probability. "
        "Read as: p_theta of x at t+1 given x up to and including t."
    ),
    r"p_i(\tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}, \qquad \tau>0.": (
        "Temperature-scaled softmax: divide logits by temperature τ before exponentiation. "
        "τ < 1 sharpens the distribution (more peaked, less diverse). "
        "τ > 1 flattens it (more uniform, more random). "
        "τ = 1 is standard softmax. τ → 0 approaches greedy argmax."
    ),
    r"\hat{x}_{t+1}=\arg\max_i p_\theta(i\mid x_{\le t}).": (
        "Greedy decoding: pick the single token with the highest probability. "
        "Deterministic but often produces repetitive or degenerate text."
    ),
    r"\log p(y_{1:i}\mid x) = \log p(y_{<i}\mid x) + \log p(y_i\mid x,y_{<i}).": (
        "Beam search scoring decomposed: the log-probability of generating tokens y_1 through y_i "
        "equals the log-probability of the prefix y_{<i} plus the log-probability of the next token y_i "
        "given the prefix. This additive property makes beam search efficient — extend beams one token at a time. "
        "Read as: log p of y_1 through y_i given x equals log p of y less than i given x, "
        "plus log p of y_i given x and y less than i."
    ),
    r"p_i = 0 \quad \text{for } i\notin S_k.": (
        "Top-k filtering: zero out all probabilities except the k highest. "
        "Then renormalize so the remaining probabilities sum to 1. "
        "This prevents the model from sampling low-probability junk tokens."
    ),

    # ── Chapter 8: Fine-Tuning ──
    r"L(\theta) = \frac{\sum_{b,t} m_{b,t}\left[-\log p_\theta(y_{b,t}\mid x_{b,\le t})\right]} {\sum_{b,t}m_{b,t}}.": (
        "Masked loss for supervised fine-tuning: only positions where m_{b,t} = 1 (response tokens) contribute to loss. "
        "Prompt positions (m = 0) are ignored. The denominator normalizes by the total number of response tokens, "
        "so the loss is an average over responses only."
    ),
    r"L(\theta) = \frac{\sum_{b,t}m_{b,t}\ell_{b,t}(\theta)}{\sum_{b,t}m_{b,t}}.": (
        "Weighted loss for fine-tuning: only positions where mask m is non-zero contribute to the loss. "
        "The numerator sums the masked losses; the denominator normalizes by the total mask weight. "
        "When m ∈ {0,1}, this is the average loss over response tokens only, ignoring prompt tokens entirely."
    ),
    r"\frac{1 + 0.5 + 0.25}{3}.": (
        "A weighted average example: (1 + 0.5 + 0.25) / 3 illustrates how loss masking with non-binary weights works "
        "during supervised fine-tuning — positions with higher weight contribute proportionally more to the loss. "
        "Read as: one plus 0.5 plus 0.25, all divided by 3."
    ),

    # ── Chapter 10: Quantization ──
    r"q \in \{q_{\min}, q_{\min}+1, \ldots, q_{\max}\}.": (
        "The quantized value q is restricted to a finite set of integers from q_min to q_max. "
        "For INT8: q ∈ {−128, −127, ..., 127}. For UINT8: q ∈ {0, 1, ..., 255}. "
        "The finite integer range is what makes quantization save memory — we store small integers instead of float32 values."
    ),
    r"q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s} + z\right), q_{\min}, q_{\max}\right),": (
        "Quantization formula: divide the real value x by scale s, add the zero-point z, "
        "round to the nearest integer, then clip to the valid integer range [q_min, q_max]. "
        "This maps continuous real values to a finite set of discrete integer levels."
    ),
    r"\hat{x} = s(q - z).": (
        "Dequantization: recover an approximate real value from the quantized integer q. "
        "Subtract the zero-point, multiply by the scale. The hat on x indicates this is an approximation — "
        "some information was lost during the round-to-integer step."
    ),
    r"q_{\min}=0,\qquad q_{\max}=255.": (
        "Unsigned 8-bit integer range: q_min = 0 and q_max = 255, giving 256 possible values (2⁸). "
        "UINT8 quantization uses this asymmetric range, requiring a zero-point to handle real values that cross zero. "
        "Read as: q_min equals 0, q_max equals 255."
    ),
    r"s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}},": (
        "Scale factor for quantization: the range of real values divided by the range of integer levels. "
        "A smaller scale means finer quantization (more precision); a larger scale means coarser steps. "
        "The scale determines how much each integer step corresponds to in real value space."
    ),
    r"z = \operatorname{round}\left(q_{\min} - \frac{x_{\min}}{s}\right),": (
        "Zero-point: the integer value that corresponds to real value zero. "
        "For symmetric quantization (INT8), z = 0 because the range is symmetric around zero. "
        "For asymmetric (UINT8), z is typically around 128 to center the range."
    ),
    r"q_{i,j} = \operatorname{round}\left(\frac{x_{i,j}}{s_i}+z_i\right).": (
        "Per-channel quantization: for each channel i, divide the real value x_{i,j} by the channel's scale s_i, "
        "add the zero-point z_i, and round to the nearest integer. "
        "Each channel gets its own scale and zero-point, making this significantly more accurate than per-tensor quantization "
        "for tensors where different channels have different value ranges."
    ),
    r"\text{bytes} = 2 \cdot L \cdot H \cdot D \cdot T \cdot B \cdot \text{bytes per value},": (
        "KV-cache memory formula: 2 (for K and V) × L layers × H heads × D head-dim × T sequence length × B batch × bytes_per_value. "
        "At FP16 (2 bytes), a 32-layer model with 32 heads, dim 128, seq 2048, batch 1 uses 2×32×32×128×2048×1×2 ≈ 1GB. "
        "Quantizing KV-cache to INT8 reduces this by roughly 4×, saving hundreds of megabytes."
    ),
    r"O(L\cdot H\cdot D\cdot T).": (
        "Standard KV-cache memory complexity: linear in all factors — layers, heads, dimension, and sequence length. "
        "The dominant term is typically T (sequence length), which grows during autoregressive generation."
    ),

    # ── Chapter 11: Modern LLM Practice ──
    r"\mathbb{E}_{x\sim \mathcal{D}_{\text{pretrain}}}\left[\sum_t -\log p_\theta(x_{t+1}\mid x_{\le t})\right].": (
        "The pretraining objective: expected value (over documents drawn from the pretraining distribution) "
        "of the sum of negative log-probabilities for each next-token prediction. "
        "The expectation E means we average over the entire data distribution, not just a single batch. "
        "Read as: expectation over x drawn from D-pretrain of the sum over t of negative log p_theta of x at t+1 given x up to t."
    ),
    r"\text{maximize reward}(x,y) - \beta\left[\log \pi_\theta(y\mid x)-\log \pi_{\mathrm{ref}}(y\mid x)\right],": (
        "The RLHF/DPO objective in intuitive form: maximize the reward for the generated response y, "
        "minus a penalty term (scaled by β) that prevents the model from drifting too far from the reference model. "
        "The penalty is the difference in log-probability between the current policy π_θ and the reference policy π_ref. "
        "Read as: maximize reward of (x,y) minus beta, times log pi-theta of y given x minus log pi-ref of y given x."
    ),
    r"\log \frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)} - \log \frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(": (
        "The DPO (Direct Preference Optimization) loss: compare how much more the trained model "
        "favors the preferred response y⁺ over the dispreferred y⁻, relative to a reference model. "
        "We want the log-ratio to be larger for preferred responses. Read as: log of pi-theta of y-plus given x "
        "over pi-ref of y-plus, minus log of pi-theta of y-minus over pi-ref of y-minus."
    ),
    r"p_\theta(x_{\text{answer}}\mid x_{\text{question}}, r_1,\ldots,r_k),": (
        "RAG generation probability: the model assigns probability to an answer conditioned on "
        "both the question and k retrieved passages r_1 through r_k. The retrieved passages "
        "are concatenated into the context before the answer, making this look like standard conditional generation. "
        "Read as: p_theta of x-answer given x-question and retrieved passages r_1 through r_k."
    ),
    r"\text{question} \to \text{retriever} \to \text{passages} \to \text{LM context} \to \text{answer}.": (
        "The RAG (Retrieval-Augmented Generation) pipeline: a question flows through a retriever "
        "which finds relevant passages, those passages are added to the LM's context window, "
        "and the LM generates an answer informed by the retrieved information. "
        "Each arrow represents one data transformation step in the pipeline."
    ),

    # ── Chapter 12: Beyond Transformers ──
    r"O(Tw)": (
        "Sliding window attention cost: each token attends to only w neighboring positions "
        "(a local window) rather than the full T positions. This gives O(T·w) cost instead of O(T²), "
        "which is dramatically cheaper for long sequences."
    ),
    r"O(T^2).": (
        "Full (dense) attention cost: quadratic in sequence length T. "
        "Every token attends to every other token. For T = 4096, that's over 16 million scores per head. "
        "This quadratic scaling is the main reason long-context models need specialized architectures."
    ),
    r"L\cdot H\cdot D\cdot T.": (
        "Full multi-head attention KV-cache: L layers × H heads × D dim × T tokens. "
        "With 32 heads and dim 128, each token adds 32×128 = 4,096 values to the cache per layer."
    ),
    r"L\cdot 1\cdot D\cdot T.": (
        "Multi-query attention (MQA) KV-cache: all heads share a single K/V pair per layer. "
        "The factor H becomes 1, meaning the cache is H times smaller than standard MHA. "
        "This dramatically reduces memory at a small cost in quality."
    ),
    r"L\cdot G\cdot D\cdot T.": (
        "Grouped-query attention (GQA) KV-cache: G groups share K/V, where 1 ≤ G ≤ H. "
        "G = H is standard MHA; G = 1 is multi-query attention. "
        "This is a sweet spot between MHA's quality and MQA's efficiency. Llama 2, Llama 3, and many modern models use GQA."
    ),
    r"\text{tokens} \to \text{embeddings} \to \text{causal transformer states} \to \text{logits} \to \text{loss or samples}.": (
        "The full data flow through a language model: raw token IDs → embedding vectors → contextual representations "
        "from transformer blocks → logit scores over the vocabulary → either a scalar loss (training) or sampled tokens (generation). "
        "Each arrow represents one stage of computation in the forward pass."
    ),

    # ── Inline symbols with meaningful readings ──
    r"y_{b,t}": (
        "y_{b,t}: the ground-truth target token ID at batch b, position t. The model should assign high probability to this token."
    ),
    r"y \in \{1,\ldots,V\}": (
        "y is in {1,...,V}: the ground-truth class label. An integer between 1 and V."
    ),
    r"p = \operatorname{softmax}(z)": (
        "p = softmax(z): convert logits z to a probability distribution p in one operation."
    ),
    r"p_y": (
        "p_y: the probability the model assigns to the correct class y. We want this to be as close to 1 as possible."
    ),
    r"p_i = \operatorname{softmax}(z)_i": (
        "p_i = softmax(z)_i: the model's predicted probability for class/token i. Always between 0 and 1."
    ),
    r"z_k": (
        "z_k: the logit (raw score) for class k. Not a probability — it can be any real number, positive or negative."
    ),
    r"L(z,y) = -z_y + \log\sum_j e^{z_j}": (
        "Loss equals −z_y plus log-sum-exp: subtract the correct logit, add the log of the denominator. "
        "This is the numerically stable form of cross-entropy used in actual implementations."
    ),
    r"\partial L/\partial z_i": (
        "∂L/∂z_i: partial derivative of the loss with respect to logit z_i. "
        "The key quantity for backpropagation — tells us how much changing each logit affects the loss."
    ),
    r"\nabla_\theta L": (
        "nabla_theta L: the gradient of loss L with respect to all parameters θ. "
        "A vector of partial derivatives, one per trainable parameter in the model."
    ),
    r"z_\theta(b,t) \in \mathbb{R}^V": (
        "z_θ(b,t) is in R^V: the logit vector at batch b, position t. V entries, one per vocabulary item."
    ),
    r"H(q)": (
        "H(q): the entropy of the true distribution q. For one-hot targets, H(q) = 0 — there is no uncertainty in the label."
    ),
    r"\mathrm{KL}(q\|p)": (
        "KL(q‖p): Kullback-Leibler divergence from p to q. Measures the extra bits needed to encode data "
        "from distribution q using the model's distribution p. Always ≥ 0, zero only when p = q."
    ),
    r"\log \sum_j e^{z_j}": (
        "log-sum-exp: the log of the sum of exponentiated logits. Appears in the softmax denominator and cross-entropy loss. "
        "This is the numerically stable way to compute the normalization constant."
    ),
    r"p_y - 1": (
        "p_y − 1: the gradient of cross-entropy with respect to the correct class's logit. Always negative (≤ 0), "
        "meaning backpropagation will increase z_y — pushing the model to assign higher probability to the correct answer."
    ),
    r"q": (
        "q: the true (target) probability distribution. For classification with one-hot labels, q_y = 1 and all other q_i = 0."
    ),
    r"\operatorname{softmax}(QK^T)V": (
        "softmax(QK^T)V: the full attention computation in one compact expression. "
        "Compute all pairwise attention scores (QK^T), apply softmax to get weights, multiply by values (V). "
        "This is the core operation that every transformer repeats in every layer."
    ),
    r"\operatorname{softmax}(z/\tau).": (
        "softmax(z/τ): temperature-scaled softmax. Divide logits by τ before softmax. "
        "τ controls the sharpness of the output distribution — low τ for precision, high τ for diversity."
    ),
    r"\Delta^{V-1}": (
        "The (V−1)-simplex: the set of all valid probability distributions over V items. "
        "Every output of softmax lies in this geometric set."
    ),
    r"\mathbb{R}^{B\times T\times C}": (
        "R^{B×T×C}: a real-valued tensor with B batches, T time steps, and C channels. "
        "This is the shape of the residual stream that flows through every transformer block."
    ),
    r"\mathcal{V}^T": (
        "Calligraphic V superscript T: the set of all possible token sequences of length T over vocabulary V. "
        "This is the sample space of the language model — every possible input and output is a point in this set."
    ),
    r"x_{<t}": (
        "x sub less-than-t: all tokens before position t — the prefix (x_1, ..., x_{t−1}). "
        "The model conditions on this entire prefix to predict x_t."
    ),
    r"\bar{x}": (
        "x-bar: the corrupted/masked version of the input. Some tokens have been replaced with [MASK], "
        "and the model must predict the originals. Used in masked language modeling (BERT-style) training."
    ),
    r"A(x)": (
        "A of x: the set of masked token positions in a masked language modeling example. "
        "Only these positions contribute to the loss during BERT-style training."
    ),
    r"\operatorname{decode}: \mathcal{V}^T \to \text{strings}.": (
        "The decode function: maps a sequence of token IDs back to a text string. "
        "Not always perfectly invertible — some tokenizers (like BPE) may lose whitespace or capitalization information."
    ),
    r"L = -z_y + \log\sum_j e^{z_j}": (
        "L = −z_y + log-sum-exp: the numerically stable cross-entropy formula used in production code. "
        "Subtract the max before computing log-sum-exp for additional stability."
    ),
    r"S_k": (
        "S_k: the set of the k highest-probability tokens in the vocabulary. "
        "All tokens outside S_k get their probability set to 0 during top-k sampling."
    ),
    r"\tau": (
        "τ (tau): the temperature parameter controlling softmax sharpness. "
        "Low τ ≈ 0.1 makes the distribution sharply peaked (near-deterministic). High τ ≈ 2.0 makes it nearly uniform (highly random)."
    ),
    r"QK^T/\sqrt{D}": (
        "QK^T / √D: scaled attention scores — the core operation of transformer attention. "
        "Matrix multiplication of queries and transposed keys, divided element-wise by the square root of head dimension."
    ),
    r"\frac{1}{BT}\sum_{b,t}-\log p_\theta(y_{b,t}\mid x_{b,\le t}).": (
        "Compact form of the batch empirical risk: average negative log-likelihood over all batch and position pairs. "
        "This is the scalar value printed as 'loss' during training. Read as: one over BT times the sum over b and t of negative log p_theta."
    ),
    r"\frac{\partial L}{\partial w_i} = \frac{2w_i}{3}.": (
        "Gradient of a simple quadratic loss L(w) = (1/3) Σ w_i²: the derivative with respect to w_i is (2/3)w_i. "
        "This warmup example shows how autograd computes partial derivatives mechanically before applying the same chain rule "
        "to the full transformer. Read as: partial derivative of L with respect to w_i equals 2w_i over 3."
    ),
    r"L(w) = \frac{1}{3}\sum_{i=1}^{3} w_i^2.": (
        "A toy scalar loss for demonstrating autograd: the average squared value of three parameters w_1, w_2, w_3. "
        "Minimizing this drives all w_i toward zero. This warmup isolates the gradient computation mechanism "
        "before applying it to the full transformer. Read as: L of w equals one-third times the sum from i equals 1 to 3 of w_i squared."
    ),
    r"z = (z_1,\ldots,z_V) \in \mathbb{R}^V.": (
        "The logit vector z has V entries, one per vocabulary item. "
        "Each z_i is a real number (positive, negative, or zero) representing the model's raw score for token i. "
        "After softmax, this becomes a proper probability distribution. "
        "Read as: z equals the tuple (z_1 through z_V), which lives in V-dimensional real space."
    ),
    r"H(w) = -\sum_s w_s\log w_s": (
        "H(w) = −Σ w_s log w_s: the entropy of the attention distribution w. "
        "Measures how spread out (uncertain) the attention is. High entropy means attention is diffuse across many positions; "
        "low entropy means attention is focused on just a few positions."
    ),
    r"\operatorname{encode}: \text{strings} \to \mathcal{V}^T,": (
        "The encode function: maps a text string to a sequence of T token IDs from the vocabulary. "
        "This is the forward direction — raw text goes in, integer sequence comes out."
    ),
    r"\Delta^{V-1} = \{p \in \mathbb{R}^{V}: p_i \ge 0,\ \sum_i p_i = 1\}.": (
        "The (V−1)-dimensional probability simplex: all vectors p of length V where each entry is non-negative "
        "and all entries sum to 1. The softmax output always lives in this set. "
        "The dimension is V−1 because the sum-to-1 constraint removes one degree of freedom. "
        "Read as: Delta to the V minus 1 equals the set of p in R to the V such that p_i is at least 0 and the sum of all p_i equals 1."
    ),
    r"\frac{\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{D}}\right) \approx 1.": (""),
    r"\frac{\operatorname{softmax}(z+c\mathbf{1})_i = \frac{e^{z_i+c}}{\sum_j e^{z_j+c}} = \frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}} = ": (""),
}


# ── merged v2 readings for the textbook-standard rewrite ──
from _eq_readings_v2 import EQ_READINGS_V2
EQ_READINGS.update(EQ_READINGS_V2)

from _eq_readings_v3 import EQ_READINGS_V3
EQ_READINGS.update(EQ_READINGS_V3)

from _eq_readings_v4 import EQ_READINGS_V4
EQ_READINGS.update(EQ_READINGS_V4)
