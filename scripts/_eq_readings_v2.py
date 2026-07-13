# ═══════════════════════════════════════════════════════════════════════
#  EQUATION READINGS V2 — entries for the textbook-standard rewrite of
#  notebook 99 (numbered Def/Prop/Thm environments, typed maps, worked
#  derivations, physics bridges). Keys are whitespace-normalized LaTeX,
#  matching the lookup in build_interactive_textbook.py.
# ═══════════════════════════════════════════════════════════════════════

EQ_READINGS_V2: dict[str, str] = {
    # ── Chapter 0 ──
    r"x_{1:T} = (x_1, x_2, \ldots, x_T), \qquad x_t \in \mathcal{V} = \{0,1,\ldots,V-1\},": (
        "A token sequence of length T. Each position's token x sub t is an element of the "
        "vocabulary: an integer between 0 and V minus 1, where V is the vocabulary size."
    ),
    r"\Delta^{V-1} = \Big\{p \in \mathbb{R}^{V}: p_i \ge 0,\ \sum_{i=0}^{V-1}p_i=1\Big\},": (
        "The probability simplex: the set of all vectors in V-dimensional space whose entries "
        "are nonnegative and sum to one. Every probability distribution over the vocabulary is "
        "one point of this set. The superscript V minus 1 records its dimension as a geometric object."
    ),
    r"p_\theta : \bigcup_{t=0}^{T_{\max}-1} \mathcal{V}^{t} \longrightarrow \mathring{\Delta}^{V-1}, \qquad x_{1:t} \longmapsto p_\theta(\,\cdot \mid x_{1:t}),": (
        "The language model as a typed map: its domain is the union of all prefix spaces "
        "(sequences of length 0 up to the maximum context minus one), and its codomain is the "
        "interior of the probability simplex — strictly positive distributions. Feed in a prefix, "
        "get out a full distribution over the next token. The ring over delta means 'interior'."
    ),
    r"P_\theta(X_{1:T}=x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).": (
        "The joint probability of the whole sequence is defined as the product of the per-position "
        "conditional probabilities: each token's probability given everything before it."
    ),
    r"p(x_t \mid x_{<t}) = \frac{P(X_{1:t} = x_{1:t})}{P(X_{1:t-1}=x_{1:t-1})}, \qquad 1\le t\le T,": (
        "How to recover conditionals from a joint distribution: the conditional probability of "
        "token t given its prefix is the probability of the length-t prefix divided by the "
        "probability of the length t-minus-1 prefix. This is just the definition of conditional probability."
    ),
    r"\prod_{t=1}^{T} \frac{P(x_{1:t})}{P(x_{1:t-1})} = P(x_{1:T})": (
        "A telescoping product: each numerator cancels the next denominator, so only the final "
        "numerator survives — the probability of the full sequence. This is why the chain-rule "
        "factorization loses no information."
    ),
    r"\log P_\theta(x_{1:T}) = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}),": (
        "Taking the logarithm turns the product into a sum: the log of a product equals the sum "
        "of the logs. Each token contributes one term."
    ),
    r"\sum_{i\in A(x)} \log p_\theta(x_i\mid \bar{x}).": (
        "Masked language modeling objective: sum the log-probabilities only over the selected "
        "subset A(x) of masked positions, with the model reading the corrupted input x-bar. "
        "This is the BERT-style contrast to causal prediction."
    ),
    r"\texttt{idx}[b,t] \in \mathcal{V}.": (
        "The input tensor entry at batch index b and position t is a vocabulary element: "
        "one integer token ID."
    ),
    r"\widehat{R}(\theta) = \frac{1}{N}\sum_{n=1}^{N}\frac{1}{T_n-1}\sum_{t=1}^{T_n-1} -\log p_\theta\big(x^{(n)}_{t+1}\mid x^{(n)}_{1:t}\big).": (
        "The empirical risk over a dataset: average over the N sequences, and within each sequence "
        "average over its positions, the negative log-probability the model gave to the actual next token. "
        "The hat on R means it is an estimate computed from data."
    ),
    r"\widehat{R}_{\text{batch}}(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta\big(y_{b,t}\mid x_{b,1:t}\big).": (
        "The batch-level version actually implemented: average the per-token negative log-likelihood "
        "over all B windows and T positions. Here y sub b,t is the shifted target — the token that "
        "actually came next. This scalar is what backpropagation differentiates."
    ),
    # ── Chapter 1 ──
    r"\Theta \;=\; \underbrace{\mathbb{R}^{V\times C}}_{W_E} \times \underbrace{\mathbb{R}^{T_{\max}\times C}}_{W_P} \times \prod_{l=1}^{L}\Big( \underbrace{\mathbb{R}^{C\times 3C}}_{W^{(l)}_{QKV}}\times \underbrace{\mathbb{R}^{C\times C}}_{W^{(l)}_{O}}\times \underbrace{\mathbb{R}^{C\times 4C}\times\mathbb{R}^{4C\times C}}_{\text{MLP}}\times \underbrace{\mathbb{R}^{C}\times\mathbb{R}^{C}\times\cdots}_{\text{LN } \gamma,\beta,\ \text{biases}} \Big) \;\cong\; \mathbb{R}^{P},": (
        "The parameter space, spelled out: a Cartesian product of one matrix space per weight tensor — "
        "the token embedding, the position table, and for each of the L layers the QKV projection, "
        "the output projection, the two MLP matrices, and the layer-norm vectors. Flattened, it is "
        "just R to the P, where P is the total parameter count. Training is a trajectory through this space."
    ),
    r"F : \Theta \times \mathcal{V}^{B\times T} \longrightarrow \mathbb{R}^{B\times T\times V}, \qquad (\theta, \texttt{idx}) \longmapsto \texttt{logits},": (
        "The model as a two-argument map: give it a parameter point theta and a batch of integer "
        "token IDs, and it returns the real-valued logit tensor of shape B by T by V. It is smooth "
        "in theta, which is what makes gradient training possible."
    ),
    r"L(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \ell\big(z_\theta(b,t),\, y_{b,t}\big),": (
        "The scalar training objective: average the per-position loss ell over every batch element "
        "and position. z sub theta of (b,t) is the logit vector at one position; y sub b,t is its target token."
    ),
    r"\ell(z,y) = -\log \operatorname{softmax}(z)_y,": (
        "Cross-entropy for one example: take the softmax of the logits, look up the probability "
        "assigned to the correct class y, and take negative log. High probability on the truth "
        "means low loss."
    ),
    r"\Theta \xrightarrow{\;F(\cdot,\,\texttt{idx})\;} \mathbb{R}^{B\times T\times V} \xrightarrow{\;\text{cross-entropy}\;} \mathbb{R}, \qquad \theta \mapsto z_\theta \mapsto L(\theta).": (
        "The training objective as a composition of two typed maps: parameters go to logits via the "
        "model, and logits go to a single real number via cross-entropy. Backpropagation walks this "
        "chain in reverse."
    ),
    r"\theta_{k+1} = \theta_k - \eta \nabla_\theta L(\theta_k), \qquad \eta>0,": (
        "Gradient descent: the next parameter point equals the current one minus the learning rate "
        "eta times the gradient of the loss. Step downhill, repeat."
    ),
    r"\dot\theta(s) = -\nabla_\theta L(\theta(s)),": (
        "The gradient flow: an ordinary differential equation whose solution moves parameters "
        "continuously in the direction of steepest descent. Gradient descent with step eta is "
        "exactly the explicit Euler discretization of this flow."
    ),
    r"\sigma: \mathbb{R}^V \longrightarrow \mathring{\Delta}^{V-1}, \qquad \sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=0}^{V-1}\exp(z_j)}.": (
        "Softmax as a typed map: from unconstrained V-dimensional score vectors to strictly positive "
        "probability vectors. Each entry is its exponential divided by the sum of all exponentials — "
        "positivity from the exponential, normalization from the division."
    ),
    r"\sigma(z+c\mathbf{1}) = \sigma(z),": (
        "Shift invariance: adding the same constant c to every logit leaves the softmax output "
        "unchanged. Only differences between logits matter."
    ),
    r"\sigma(z+c\mathbf{1})_i = \frac{e^{z_i+c}}{\sum_j e^{z_j+c}} = \frac{e^c\, e^{z_i}}{e^c \sum_j e^{z_j}} = \sigma(z)_i.": (
        "The one-line proof of shift invariance: the factor e to the c appears in both numerator "
        "and denominator and cancels exactly."
    ),
    r"\sigma(z)_i = \frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)}, \qquad m = \max_j z_j,": (
        "The numerically stable evaluation: subtract the maximum logit m from every entry before "
        "exponentiating. By shift invariance the answer is identical, but now the largest exponent "
        "is e to the zero equals one, so nothing overflows."
    ),
    r"A: \mathbb{R}^V \to \mathbb{R}, \qquad A(z) = \log \sum_{j=0}^{V-1} e^{z_j}.": (
        "The log-sum-exp, or log-partition function: exponentiate every logit, sum, take the log. "
        "In statistical mechanics the sum inside is the partition function Z, and A equals log Z — "
        "the free energy up to sign."
    ),
    r"A(z) = m + \log \sum_j e^{z_j-m}, \qquad m=\max_j z_j,": (
        "The stable identity for log-sum-exp: pull the maximum m out front, subtract it inside. "
        "This is what torch dot logsumexp implements."
    ),
    r"\nabla A(z) = \sigma(z), \qquad \nabla^2 A(z) = \operatorname{diag}(p) - pp^{\top}, \quad p = \sigma(z),": (
        "The key structural fact: softmax is the gradient of the log-partition function, and the "
        "Hessian of A — equivalently the Jacobian of softmax — is the diagonal matrix of probabilities "
        "minus the rank-one outer product p p-transpose."
    ),
    r"\frac{\partial \sigma(z)_i}{\partial z_k} = p_i(\delta_{ik} - p_k).": (
        "The softmax Jacobian in coordinates: the derivative of output i with respect to input k is "
        "p sub i times, delta i k minus p sub k, where delta is one when i equals k and zero otherwise. "
        "Raising one logit raises its own probability and drains the others."
    ),
    r"\frac{\partial A}{\partial z_i} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i = \sigma(z)_i .": (
        "Differentiating log-sum-exp with respect to one logit gives that logit's exponential over "
        "the total sum — precisely the softmax probability. Gradient of A equals softmax, proved."
    ),
    r"\frac{\partial p_i}{\partial z_k} = \frac{\delta_{ik}e^{z_i} Z - e^{z_i} e^{z_k}}{Z^2} = p_i\delta_{ik} - p_i p_k = p_i(\delta_{ik} - p_k),": (
        "The quotient-rule derivation of the softmax Jacobian: differentiate e to the z-i over Z, "
        "using that Z's derivative with respect to z-k is e to the z-k, then simplify to "
        "p-i times delta-i-k minus p-k."
    ),
    r"u^\top\big(\operatorname{diag}(p)-pp^\top\big)u = \sum_i p_i u_i^2 - \Big(\sum_i p_i u_i\Big)^2 = \operatorname{Var}_{i\sim p}(u_i) \;\ge\; 0,": (
        "Positive semidefiniteness of the Hessian, via a probabilistic identity: the quadratic form "
        "equals the expected square minus the squared expectation of u under the distribution p — "
        "that is, a variance, which can never be negative. Hence A is convex."
    ),
    r"\max_{p\,\in\,\Delta^{V-1}}\; \Big\{ \langle p, z\rangle + \tau H(p) \Big\}, \qquad H(p) = -\sum_i p_i\log p_i,": (
        "The Gibbs variational problem: over all probability vectors p, maximize the expected score "
        "(the inner product of p with z) plus tau times the Shannon entropy of p. Tau is a temperature: "
        "it sets the exchange rate between chasing high scores and staying spread out."
    ),
    r"\mathcal{L}(p,\mu) = \sum_i p_i z_i - \tau \sum_i p_i \log p_i + \mu\Big(\sum_i p_i - 1\Big).": (
        "The Lagrangian for the Gibbs problem: the objective plus a multiplier mu enforcing that "
        "the probabilities sum to one. Positivity constraints are inactive at an interior optimum, "
        "so this single multiplier suffices."
    ),
    r"z_i - \tau(\log p_i + 1) + \mu = 0 \quad\Longrightarrow\quad p_i = \exp\!\Big(\frac{z_i}{\tau}\Big)\,\exp\!\Big(\frac{\mu-\tau}{\tau}\Big).": (
        "Stationarity of the Lagrangian: set the derivative with respect to each p-i to zero and "
        "solve. The result is proportional to e to the z-i over tau; the second exponential is a "
        "constant fixed by normalization. The optimum is softmax at temperature tau."
    ),
    r"\langle p^\ast, z\rangle + \tau H(p^\ast) = \sum_i p^\ast_i z_i - \tau\sum_i p^\ast_i\Big(\frac{z_i}{\tau} - A(z/\tau)\Big) = \tau A(z/\tau). \qquad\blacksquare": (
        "Substituting the Gibbs solution back into the objective: the score terms cancel and only "
        "tau times the log-partition function at temperature tau survives. The optimal value of the "
        "variational problem is the free energy. The black square closes the proof."
    ),
    r"\ell : \mathbb{R}^V \times \mathcal{V} \longrightarrow \mathbb{R}_{\ge 0}, \qquad \ell(z,y) = -\log \sigma(z)_y .": (
        "Cross-entropy as a typed map: it eats a logit vector and a class label, and returns a "
        "nonnegative real number — minus the log of the probability softmax assigned to the true class."
    ),
    r"\ell(z,y) = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = A(z) - z_y .": (
        "Expanding the definition: the loss equals the log-partition function A of z minus the logit "
        "of the correct class. This 'gather one logit, subtract from log-sum-exp' form is what stable "
        "implementations compute, and it makes positivity obvious since A of z is at least z-y."
    ),
    r"\nabla_z\, \ell(z,y) = p - e_y,": (
        "The gradient of cross-entropy is the residual: predicted distribution p minus the one-hot "
        "vector of the true class. The correct class gets pushed up by one minus its probability; "
        "every wrong class gets pushed down by exactly the probability it wrongly received."
    ),
    r"H(q,p) = -\sum_i q_i \log p_i,": (
        "Cross-entropy between two distributions: the expected negative log-probability under the "
        "model p when outcomes are drawn from the target q. In coding terms, the average message "
        "length when you code q-data with a p-codebook."
    ),
    r"H(q,p) = H(q) + \mathrm{KL}(q\,\|\,p),": (
        "The fundamental decomposition: cross-entropy equals the entropy of the target plus the "
        "Kullback–Leibler divergence from target to model. The first term is out of the model's "
        "control; only the KL term can be reduced by training."
    ),
    r"R(\theta) := \mathbb{E}_{x\sim q}\Big[-\log p_\theta(x_t \mid x_{<t})\Big] = \underbrace{\mathbb{E}_{x_{<t}}\big[H\big(q(\cdot\mid x_{<t})\big)\big]}_{\text{irreducible}} \;+\; \underbrace{\mathbb{E}_{x_{<t}}\Big[\mathrm{KL}\big(q(\cdot\mid x_{<t})\,\big\|\,p_\theta(\cdot\mid x_{<t})\big)\Big]}_{\ge 0,\ =0 \text{ iff } p_\theta = q}.": (
        "The population risk splits into two labelled pieces: the average intrinsic entropy of "
        "language given each prefix — the irreducible floor no model can beat — plus the average "
        "KL divergence between the true conditionals and the model's, which is nonnegative and "
        "vanishes exactly when the model matches the data distribution. Maximum likelihood is "
        "KL minimization in disguise."
    ),
    r"\mathbb{E}_{x_t}\big[-\log p_\theta(x_t\mid x_{<t})\big] = -\sum_i q(i\mid x_{<t})\log p_\theta(i\mid x_{<t}) = H\big(q(\cdot\mid x_{<t}),\, p_\theta(\cdot\mid x_{<t})\big),": (
        "The inner step of the proof: averaging the per-token loss over the true next-token "
        "distribution q produces exactly the cross-entropy between the true conditional and the "
        "model conditional at that prefix."
    ),
    r"\frac{1}{BT}\sum_{b,t}-\log p_\theta\big(y_{b,t}\mid x_{b,1:t}\big).": (
        "The implemented scalar loss: average negative log-likelihood over all batch elements and "
        "positions, with y the shifted targets."
    ),
    # ── Chapter 2 ──
    r"\operatorname{enc}: \mathcal{A}^* \longrightarrow \mathcal{V}^*, \qquad \operatorname{dec}: \mathcal{V}^* \longrightarrow \mathcal{A}^*,": (
        "A tokenizer is a pair of maps between the two sequence worlds: encode takes strings over "
        "the raw alphabet A to token-ID sequences, and decode goes back."
    ),
    r"\operatorname{dec} \circ \operatorname{enc} = \operatorname{id}_{\mathcal{A}^*}.": (
        "Losslessness: decoding after encoding returns the original string exactly, for every string. "
        "Note the composition in the other order is generally not the identity — several token "
        "sequences can decode to the same string."
    ),
    # ── Chapter 3 ──
    r"E: \mathcal{V} \longrightarrow \mathbb{R}^C,": (
        "The token embedding as a typed map: from the finite vocabulary into C-dimensional real "
        "space. Discrete symbols acquire coordinates."
    ),
    r"E^{\otimes T}: \mathcal{V}^T \longrightarrow \mathbb{R}^{T\times C}, \qquad \big(E^{\otimes T}(x)\big)_{t,:} = E(x_t).": (
        "The embedding applied position-by-position to a whole sequence: a length-T sequence of "
        "token IDs becomes a T-by-C matrix whose row t is the embedding of token x-t."
    ),
    r"E = (\,\cdot\, W_E)\circ \iota, \qquad e_i W_E = W_E[i,:].": (
        "The embedding factors as one-hot inclusion followed by matrix multiplication: send token i "
        "to the standard basis vector e-i, then multiply by the embedding matrix, which picks out "
        "row i. Lookup is one-hot matrix multiplication, implemented efficiently."
    ),
    r"\mathrm{In}: \mathcal{V}^T \longrightarrow \mathbb{R}^{T\times C}, \qquad \mathrm{In}(x)_{t,:} = W_E[x_t,:] + W_P[t,:],": (
        "The input map of the model: row t of the initial residual stream is the token embedding of "
        "x-t plus the position embedding of t. Token identity and position enter the same "
        "C-dimensional space additively — and the position term is what breaks permutation symmetry."
    ),
    r"(x_1,x_2,\ldots,x_{T+1}) \in \mathcal{V}^{T+1}": (
        "A raw window of T plus one consecutive tokens, the raw material for T prediction examples."
    ),
    r"(x_1) \mapsto x_2,\qquad (x_1,x_2) \mapsto x_3,\qquad \ldots,\qquad (x_1,\ldots,x_T) \mapsto x_{T+1}.": (
        "The T supervised examples hiding in one window: each prefix maps to the token that follows it, "
        "from the single-token prefix up to the full length-T prefix."
    ),
    r"\mathcal{V}^{T+1} \longrightarrow \mathcal{V}^{T}\times\mathcal{V}^{T}, \qquad x_{1:T+1} \longmapsto \big(\underbrace{x_{1:T}}_{\text{input}},\ \underbrace{x_{2:T+1}}_{\text{target}}\big),": (
        "Label shifting as a typed map: a window of T plus one tokens becomes an input-target pair "
        "of length-T sequences — the input is the first T tokens, the target is the same window "
        "shifted left by one."
    ),
    r"L(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta\big(y_{b,t}\mid x_{b,1:t}\big).": (
        "The batch loss from shifted labels: at every batch element and position, the negative "
        "log-probability of the next token given the prefix up to and including position t — "
        "the conditioning the causal mask enforces."
    ),
    # ── Chapter 4 ──
    r"M \in \{0,-\infty\}^{T\times T}, \qquad M_{t,s} = \begin{cases} 0, & s\le t,\\ -\infty, & s>t, \end{cases}": (
        "The additive causal mask: a T-by-T matrix that is zero where source position s is at or "
        "before receiving position t, and minus infinity where s is in the future. Added to scores "
        "before softmax, minus infinity becomes probability zero."
    ),
    r"\mathrm{A}: \mathbb{R}^{T\times D}\times\mathbb{R}^{T\times D}\times\mathbb{R}^{T\times D} \longrightarrow \mathbb{R}^{T\times D}, \qquad \mathrm{A}(Q,K,V) = W\,V,": (
        "Single-head attention as a typed map: three T-by-D matrices in — queries, keys, values — "
        "one T-by-D matrix out, computed as a weight matrix W times the values. All the nonlinearity "
        "lives in how W is built from Q and K."
    ),
    r"A = QK^{\top} \in \mathbb{R}^{T\times T} \quad\text{(scores, } a_{t,s} = q_t\cdot k_s\text{)},": (
        "The score matrix: Q times K-transpose, a T-by-T matrix whose entry at row t, column s is "
        "the dot product of query t with key s — how well position t's question matches position s's address."
    ),
    r"W_{t,:} = \operatorname{softmax}\!\left(\frac{A_{t,:}}{\sqrt{D}} + M_{t,:}\right) \in \Delta^{T-1} \quad\text{(row-wise, } t = 1,\ldots,T\text{)},": (
        "Each row of scores is divided by root D (the temperature), the causal mask is added, and "
        "softmax turns the row into a probability distribution over source positions. Every row of "
        "W is a point of the simplex."
    ),
    r"o_t = \sum_{s=1}^{T} W_{t,s}\, v_s \quad\text{(output rows)}.": (
        "The output at position t is the weighted average of the value vectors, weighted by "
        "position t's attention distribution."
    ),
    r"\mathcal{S}_T = \Big\{ W\in\mathbb{R}^{T\times T} :\; W_{t,s}\ge 0,\ \ \sum_{s=1}^{T} W_{t,s}=1,\ \ W_{t,s}=0 \text{ for } s>t \Big\},": (
        "The space of causal row-stochastic matrices: nonnegative entries, every row summing to one, "
        "and zeros above the diagonal. Every causal attention weight matrix lives here — this set is "
        "the true codomain of the score-to-weight computation."
    ),
    r"(Q,K) \longmapsto W(Q,K) \in \mathcal{S}_T, \qquad (W, V) \longmapsto WV,": (
        "Attention factors into two stages: queries and keys nonlinearly produce a causal stochastic "
        "matrix, and that matrix acts linearly on the values. Attention is linear in V, nonlinear in the data overall."
    ),
    r"o_t \in \operatorname{conv}\{v_1,\ldots,v_t\} \subset \mathbb{R}^D.": (
        "The geometric constraint: each output vector lies in the convex hull of the value vectors "
        "of its own prefix. A head can mix and route what it sees, but never leave that hull — "
        "escaping it is the job of the residual connection and the MLP."
    ),
    r"\mathrm{A}_0(P_\pi X) = P_\pi\, \mathrm{A}_0(X) \qquad \text{for all } X \in \mathbb{R}^{T\times C},\ \pi\in S_T.": (
        "Permutation equivariance: reordering the input rows and then applying unmasked attention "
        "gives the same result as applying attention first and reordering after. The layer has no "
        "idea where anything is — which is precisely why position embeddings must exist."
    ),
    r"(P_\pi Q)(P_\pi K)^{\top} = P_\pi\, QK^{\top} P_\pi^{\top}.": (
        "Step one of the proof: permuting the rows of Q and K conjugates the score matrix — rows and "
        "columns get permuted together."
    ),
    r"(P_\pi W P_\pi^{\top})(P_\pi V) = P_\pi (WV),": (
        "Step three of the proof: the conjugated weight matrix times the permuted values equals the "
        "permuted output, using that P-transpose P is the identity. Equivariance follows."
    ),
    r"\max_{w\,\in\,\Delta^{t-1}}\ \Big\{ \textstyle\sum_{s\le t} w_s a_s \;+\; \tau H(w) \Big\}": (
        "The attention row as a variational problem: over distributions w on the prefix positions, "
        "maximize expected score plus tau times entropy. Same Gibbs problem as the softmax head, "
        "now over positions instead of vocabulary."
    ),
    r"\mathcal{L}(w,\mu) = \sum_s w_s a_s - \tau\sum_s w_s\log w_s + \mu\Big(\sum_s w_s - 1\Big),": (
        "The Lagrangian for the attention version of the Gibbs problem — identical in form to the "
        "softmax one, with scores a and multiplier mu enforcing normalization."
    ),
    # ── Lemma 4.7 ──
    r"\mathbb{E}[\,q\cdot k\,]=0, \qquad \operatorname{Var}(q\cdot k) = D, \qquad \operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt{D}}\right) = 1 .": (
        "The variance lemma: under the initialization model (independent, zero-mean, unit-variance "
        "coordinates), a D-dimensional dot product has mean zero and variance D — so dividing by "
        "root D restores variance one, keeping the attention temperature independent of head width."
    ),
    r"\mathbb{E}[(q_rk_r)^2] = \mathbb{E}[q_r^2]\,\mathbb{E}[k_r^2] = 1\cdot 1 = 1,": (
        "Each coordinate product has second moment one: by independence the expectation factorizes "
        "into the two unit second moments."
    ),
    r"\mathbb{E}[q_rk_r\,q_{r'}k_{r'}] = \mathbb{E}[q_rq_{r'}]\,\mathbb{E}[k_rk_{r'}] = 0 .": (
        "Distinct coordinate products are uncorrelated: the cross-expectation factorizes and each "
        "factor is zero. So the D variances simply add, giving variance D."
    ),
    r"\operatorname{softmax}\big((\ldots, a, \ldots, -\infty, \ldots)\big) = \big(\ldots, \tfrac{e^{a}}{Z}, \ldots, 0, \ldots\big),": (
        "What minus infinity does under softmax: e to the minus infinity is zero, so masked "
        "positions receive exactly zero probability and the rest renormalize among themselves."
    ),
    # ── Chapter 5 ──
    r"\operatorname{LN}: \mathbb{R}^C \longrightarrow \mathbb{R}^C, \qquad \operatorname{LN}(u)_i = \gamma_i\,\frac{u_i-\mu(u)}{\sqrt{\sigma^2(u)+\epsilon}} + \beta_i,": (
        "Layer norm as a typed map on one token vector: subtract the mean across channels, divide "
        "by the standard deviation (epsilon guards against division by zero), then apply the learned "
        "per-channel scale gamma and shift beta."
    ),
    r"\mu(u) = \frac{1}{C}\sum_{i=1}^{C} u_i, \qquad \sigma^2(u) = \frac{1}{C}\sum_{i=1}^{C}\big(u_i-\mu(u)\big)^2 .": (
        "The per-token statistics: mu is the mean of the C channel values of this one token vector, "
        "sigma squared their variance. Each position is normalized independently."
    ),
    r"S = \Big\{ v\in\mathbb{R}^C : \textstyle\sum_i v_i = 0,\ \ \lVert v\rVert_2 = \sqrt{C} \Big\},": (
        "The image of the normalization step: vectors with zero mean and Euclidean norm root C — "
        "a sphere of radius root C inside the mean-zero hyperplane, of dimension C minus 2. "
        "Layer norm quotients out exactly two degrees of freedom: overall offset and overall scale."
    ),
    r"\operatorname{MLP}: \mathbb{R}^C \to \mathbb{R}^C, \qquad \operatorname{MLP}(u) = W_2\,\operatorname{GELU}(W_1u + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{4C\times C},\ W_2\in\mathbb{R}^{C\times 4C},": (
        "The positionwise feed-forward map: expand from C to 4C dimensions, apply the GELU "
        "nonlinearity, project back to C. Applied to every token vector independently."
    ),
    r"\operatorname{GELU}(u) = u\,\Phi(u), \qquad \Phi(u) = \tfrac{1}{2}\Big(1 + \operatorname{erf}\big(u/\sqrt{2}\big)\Big),": (
        "GELU is the input times the standard normal cumulative distribution function Phi — a smooth "
        "ramp that is nearly zero for very negative inputs and nearly the identity for large positive "
        "ones: a mollified ReLU."
    ),
    r"x' = x + \operatorname{Attn}(\operatorname{LN}_1(x)), \qquad \mathcal{B}(x) = x' + \operatorname{MLP}(\operatorname{LN}_2(x')),": (
        "The pre-norm transformer block: normalize, attend, add back to the stream; then normalize, "
        "apply the MLP, add back again. Both updates preserve the T-by-C shape, which is what lets "
        "blocks stack."
    ),
    r"\frac{\partial(x + f(x))}{\partial x} = I + J_f(x),": (
        "The Jacobian of a residual update is the identity plus the Jacobian of the sublayer. "
        "That identity term is the whole story of trainable depth."
    ),
    r"\frac{\partial x_L}{\partial x_0} = \prod_{l=L}^{1}\big(I + J_{f_l}(x_{l-1})\big) = I + \sum_l J_{f_l} + \text{(higher products)} .": (
        "Through L residual layers, the end-to-end Jacobian is a product of identity-plus-perturbation "
        "factors, which expands to identity plus the sum of the individual Jacobians plus higher-order "
        "products. The leading identity is a direct gradient highway from output to input — no "
        "exponential attenuation, unlike a bare product of small factors."
    ),
    r"\dot x(t) = f(x(t), t),": (
        "The continuous-time reading: a residual stack is the explicit Euler discretization, with "
        "unit step, of this flow — each token vector follows a trajectory driven alternately by "
        "attention (coupling positions) and the MLP (acting pointwise)."
    ),
    r"\mathbb{R}^C \cong \bigoplus_{h=1}^{H} \mathbb{R}^{D},": (
        "Multi-head structure: the channel space splits as a direct sum of H subspaces of dimension "
        "D each; every head runs its own attention in its own subspace, and the results are "
        "concatenated back and mixed by the output projection."
    ),
    r"\mathcal{V}^{T} \;\xrightarrow{\ \mathrm{In}\ (\text{Def } 3.2)\ }\; \mathbb{R}^{T\times C} \;\xrightarrow{\ \mathcal{B}_1\ }\; \mathbb{R}^{T\times C} \;\xrightarrow{\ \mathcal{B}_2\ }\; \cdots \;\xrightarrow{\ \mathcal{B}_L\ }\; \mathbb{R}^{T\times C} \;\xrightarrow{\ \operatorname{LN}_f\ }\; \mathbb{R}^{T\times C} \;\xrightarrow{\ \cdot\,W_U\ }\; \mathbb{R}^{T\times V} \;\xrightarrow{\ \sigma\ (\text{rows})\ }\; \big(\mathring{\Delta}^{V-1}\big)^{T},": (
        "The whole model in one typed line: token IDs are embedded into the residual stream, pass "
        "through L shape-preserving blocks, get a final layer norm, are multiplied by the unembedding "
        "matrix into logit space, and each row goes through softmax into the simplex. Row t of the "
        "output is the conditional distribution for the token after position t — one forward pass, "
        "all T conditionals at once."
    ),
    r"z = h\,W_E^{\top} \in \mathbb{R}^{V}, \qquad z_i = \langle h,\ E(i)\rangle .": (
        "Weight tying, precisely: with the head sharing the embedding matrix, the logit of token i "
        "is the inner product of the final hidden state h with token i's own embedding vector. "
        "Scoring is matching in embedding space."
    ),
    r"p_\theta(i \mid x_{1:t}) = \frac{e^{\langle h_t, E(i)\rangle}}{\sum_j e^{\langle h_t, E(j)\rangle}}": (
        "The tied output distribution is a Gibbs distribution whose energies are negative "
        "context-token matching scores: the better the hidden state aligns with a token's embedding, "
        "the more probability that token receives."
    ),
    # ── Chapter 6 ──
    r"L(\theta) = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta\big(y_{b,t}\mid x_{b,1:t}\big),": (
        "The training objective once more: mean negative log-likelihood of the shifted targets over "
        "the batch — the function whose gradient the optimizer follows."
    ),
    r"g_S(\theta) = \frac{1}{m}\sum_{n\in S}\nabla \ell_n(\theta).": (
        "The minibatch gradient estimator: average the per-example gradients over a random subset S "
        "of size m instead of the whole dataset."
    ),
    r"\mathbb{E}_S\big[g_S(\theta)\big] = \frac{1}{m}\sum_{n=1}^{N}\Pr[n\in S]\,\nabla\ell_n(\theta) = \frac{1}{m}\cdot\frac{m}{N}\sum_{n=1}^{N}\nabla\ell_n(\theta) = \nabla\widehat{R}(\theta). \qquad\blacksquare": (
        "Unbiasedness in three steps: each example lands in the batch with probability m over N, so "
        "by linearity of expectation the estimator's mean is exactly the full empirical gradient. "
        "Batch size controls variance, never bias."
    ),
    r"m_k = \beta_1 m_{k-1} + (1-\beta_1)\, g_k \qquad\text{(first-moment EMA, } m_0=0\text{)},": (
        "AdamW line one: an exponential moving average of gradients — momentum. Beta-one close to "
        "one means a long memory."
    ),
    r"v_k = \beta_2 v_{k-1} + (1-\beta_2)\, g_k^{2} \qquad\text{(second-moment EMA, } v_0=0\text{)},": (
        "AdamW line two: an exponential moving average of squared gradients, tracked per coordinate — "
        "an estimate of each parameter's recent gradient scale."
    ),
    r"\hat m_k = \frac{m_k}{1-\beta_1^{k}}, \qquad \hat v_k = \frac{v_k}{1-\beta_2^{k}} \qquad\text{(bias corrections)},": (
        "The bias corrections: because both averages start at zero, early estimates are shrunk by "
        "exactly the factor one minus beta to the k; dividing by it undoes the cold start exactly."
    ),
    r"\theta_{k+1} = \theta_k - \eta\left(\frac{\hat m_k}{\sqrt{\hat v_k}+\varepsilon} \;+\; \lambda\,\theta_k\right).": (
        "The AdamW update: smoothed gradient divided coordinatewise by its root-mean-square scale — "
        "a diagonal preconditioning that makes every coordinate's step size about eta — plus decoupled "
        "weight decay lambda theta, applied outside the preconditioner. The decoupling is the W in AdamW."
    ),
    r"\operatorname{PPL} := e^{L} = \exp\Big(\frac{1}{N}\sum_n \log \frac{1}{p_n}\Big) = \Big(\prod_{n=1}^{N} \frac{1}{p_n}\Big)^{1/N},": (
        "Perplexity is the exponential of the average loss, which rewrites as the geometric mean of "
        "the inverse predicted probabilities: an effective branching factor. Perplexity twenty means "
        "the model is on average as uncertain as a uniform choice among twenty continuations."
    ),
    # ── Chapter 7 ──
    r"p(\tau) = \sigma(z/\tau) \in \mathring\Delta^{V-1}, \qquad \tau>0 .": (
        "The temperature family: divide the logits by tau before softmax. This traces out the Gibbs "
        "family of distributions — sharper below temperature one, flatter above."
    ),
    r"\hat{x}_{t+1}=\arg\max_i\; p_\theta(i\mid x_{1:t}).": (
        "Greedy decoding: always take the single most probable token — the zero-temperature limit "
        "of the Gibbs family."
    ),
    r"\log p(y_{1:i}\mid x) = \log p(y_{<i}\mid x) + \log p(y_i\mid x,y_{<i}).": (
        "The running score in beam search: the log-probability of a partial sequence extends "
        "additively — previous total plus the new token's conditional log-probability."
    ),
    r"p(i \mid i \in S) = \frac{p_i\,\mathbf{1}[i\in S]}{\sum_{j\in S} p_j},": (
        "Truncated sampling, stated honestly: top-k and top-p both sample from the model's own "
        "distribution conditioned on the token falling in the allowed set S — zero out everything "
        "outside S, renormalize what remains."
    ),
    # ── Chapter 8 ──
    r"L_m(\theta) = \frac{\sum_{b,t} m_{b,t}\,\big[-\log p_\theta(y_{b,t}\mid x_{b,1:t})\big]} {\sum_{b,t} m_{b,t}},": (
        "The masked empirical risk: multiply each per-token loss by its mask value (one for response "
        "tokens, zero for prompt tokens) and divide by the number of unmasked positions. The "
        "denominator stops long responses from dominating just by length."
    ),
    r"\mu_m = \frac{\sum_{b,t} m_{b,t}\,\delta_{(b,t)}}{\sum_{b,t} m_{b,t}}": (
        "The same masking seen as a change of measure: a reweighted empirical distribution over token "
        "positions, uniform on the unmasked ones and zero elsewhere. Masking changes the measure, "
        "not the model or the pointwise loss."
    ),
    # ── Chapter 10 ──
    r"Q_{s,z}: \mathbb{R} \longrightarrow \mathcal{Q}, \qquad Q_{s,z}(x) = \operatorname{clip}\!\Big(\operatorname{round}\Big(\frac{x}{s} + z\Big),\, q_{\min},\, q_{\max}\Big),": (
        "The affine quantizer as a typed map from the reals to a finite integer set: divide by the "
        "scale s, shift by the zero-point z, round to the nearest integer, clip into the allowed range."
    ),
    r"D_{s,z}: \mathcal{Q} \longrightarrow \mathbb{R}, \qquad D_{s,z}(q) = s\,(q-z),": (
        "The dequantizer: undo the shift and rescale. Its image is an arithmetic grid of spacing s — "
        "the finite lattice that replaces the continuum."
    ),
    r"x \in \big[\, s(q_{\min}-z),\; s(q_{\max}-z) \,\big],": (
        "The no-clipping hypothesis: x lies inside the representable interval, between the real "
        "values of the smallest and largest integer levels."
    ),
    r"|x - \hat{x}| \;\le\; \frac{s}{2}.": (
        "The half-step error bound: inside the representable range, the reconstruction error is at "
        "most half a grid spacing. Outside it, clipping makes the error grow without bound — the "
        "entire rounding-versus-clipping tradeoff lives in the single parameter s."
    ),
    r"\big|x - s(q - z)\big| = \big|x - \hat x\big| \le \frac{s}{2}.": (
        "The proof in one line: rounding to the nearest integer is off by at most one half; "
        "multiplying through by the scale s converts that to half a grid spacing in real units."
    ),
    r"s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}, \qquad z = \operatorname{round}\!\Big(q_{\min} - \frac{x_{\min}}{s}\Big),": (
        "The calibration rule: stretch the integer range across the observed real range to get the "
        "scale, then choose the zero-point so real zero lands on an integer level."
    ),
    r"\text{bytes} = 2 \cdot L \cdot H \cdot D \cdot T \cdot B \cdot (\text{bytes per value}),": (
        "The KV-cache size as a pure shape product: two (keys and values) times layers times heads "
        "times head dimension times cached sequence length times batch size times bytes per number."
    ),
    r"O(L\cdot G\cdot D\cdot T), \qquad 1\le G\le H,": (
        "Grouped-query attention cache cost: G key-value groups replace the H in the head factor. "
        "G equal to H is ordinary multi-head; G equal to one is multi-query. A memory dial, not a "
        "change to the loss."
    ),
    # ── Chapter 11 ──
    r"\mathbb{E}_{x\sim \mathcal{D}_{\text{pretrain}}}\Big[\sum_t -\log p_\theta(x_{t+1}\mid x_{1:t})\Big],": (
        "The pretraining objective at population scale: expected negative log-likelihood of next "
        "tokens over the pretraining distribution. Same objective as the toy model — what changes is scale."
    ),
    r"\max_{\pi}\; \mathbb{E}_{y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] \;-\; \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\big\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big).": (
        "The RLHF objective: choose a policy that maximizes expected reward while paying a penalty, "
        "at rate beta, for KL divergence from the reference policy. Beta is a leash length: small "
        "beta chases reward, large beta stays near the reference."
    ),
    r"\pi^{\ast}(y\mid x) = \frac{1}{Z(x)}\;\pi_{\mathrm{ref}}(y\mid x)\; \exp\!\Big(\frac{r(x,y)}{\beta}\Big), \qquad Z(x) = \sum_{y}\pi_{\mathrm{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}.": (
        "The closed-form optimum: the reference policy exponentially tilted by reward over beta, "
        "normalized by the partition function Z. A Gibbs distribution on whole responses — the "
        "reference is the prior, the reward is negative energy, beta is temperature."
    ),
    r"\sum_y \pi(y)\,r(x,y) - \beta \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)} = \sum_y \pi(y)\Big[r(x,y) + \beta\log\pi_{\mathrm{ref}}(y)\Big] + \beta H(\pi),": (
        "The reduction step: expand the KL term and absorb the reference log-probability into the "
        "score. What remains is exactly the Gibbs variational objective of Chapter 1 — expected "
        "score plus beta times entropy — now over responses."
    ),
    r"\pi^\ast(y) = \operatorname{softmax}(z/\beta)_y \;\propto\; e^{r(x,y)/\beta}\, \pi_{\mathrm{ref}}(y),": (
        "Applying the Gibbs theorem with those scores: the optimal policy is a softmax at temperature "
        "beta, proportional to the reference weighted by the exponentiated reward."
    ),
    r"r(x,y) = \beta \log\frac{\pi^{\ast}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x),": (
        "The inversion at the heart of DPO: solve the Gibbs formula for the reward in terms of the "
        "optimal policy. The intractable log Z depends only on the prompt — which is exactly why it "
        "will cancel between two responses to the same prompt."
    ),
    r"\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x,y^{+},y^{-})}\left[ \log \sigma\!\Big( \beta \log\frac{\pi_\theta(y^{+}\mid x)}{\pi_{\mathrm{ref}}(y^{+}\mid x)} - \beta \log\frac{\pi_\theta(y^{-}\mid x)}{\pi_{\mathrm{ref}}(y^{-}\mid x)} \Big)\right],": (
        "The DPO loss: a logistic loss on the margin between how much the trained policy upweights "
        "the preferred response versus the rejected one, both measured relative to the reference. "
        "No explicit reward model — the policy log-ratios play that role, and log Z has cancelled."
    ),
    r"p_\theta\big(x_{\text{answer}}\mid x_{\text{question}}, r_1,\ldots,r_k\big),": (
        "Retrieval-augmented generation: the model samples the answer conditioned on the question "
        "plus k retrieved passages inserted into the context. Retrieval changes the conditioning, "
        "not the weights."
    ),
    # ── Chapter 13 ──
    r"\mathcal{V}^{T} \xrightarrow{\ \mathrm{In}\ } \mathbb{R}^{T\times C} \xrightarrow{\ \mathcal{B}_1\ } \cdots \xrightarrow{\ \mathcal{B}_L\ } \mathbb{R}^{T\times C} \xrightarrow{\ \operatorname{LN}_f\ } \mathbb{R}^{T\times C} \xrightarrow{\ \cdot W_E^{\top}\ } \mathbb{R}^{T\times V} \xrightarrow{\ \sigma\ } \big(\mathring{\Delta}^{V-1}\big)^{T},": (
        "The closing summary of the entire model as one typed composition: embed, flow through L "
        "residual blocks, normalize, unembed against the tied embedding matrix, softmax each row "
        "into the simplex. Everything else in the book is commentary on one arrow of this line."
    ),
}
