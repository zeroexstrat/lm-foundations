#!/usr/bin/env python3
"""Build a self-contained interactive HTML textbook from notebook 99.

All math is pre-rendered with KaTeX (server-side). All code is pre-highlighted
with Pygments. All CSS, fonts, and JS are inlined. Zero external dependencies.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from _eq_readings import EQ_READINGS
from _viz_js import VIZ_CONTAINERS, VIZ_JS
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, TextLexer, get_lexer_by_name

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "99_complete_college_level_walkthrough.ipynb"
OUTPUT_PATH = ROOT / "output" / "interactive_textbook.html"
KATEX_DIR = ROOT / "node_modules" / "katex" / "dist"


# ═══════════════════════════════════════════════════════════════════════
#  1. NOTEBOOK LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_notebook() -> list[dict]:
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    return [
        {"type": c["cell_type"], "source": "".join(c["source"])}
        for c in nb["cells"]
    ]


# ═══════════════════════════════════════════════════════════════════════
#  2. MATH PRE-RENDERING (batch KaTeX via node)
# ═══════════════════════════════════════════════════════════════════════

def get_katex_js_b64() -> str:
    """Base64-encode the KaTeX JS library for inline embedding."""
    with open(KATEX_DIR / "katex.min.js", "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_math(text: str) -> tuple[str, dict[str, tuple[str, bool]]]:
    """Extract $$...$$ and $...$ math, replace with placeholders."""
    placeholders: dict[str, tuple[str, bool]] = {}
    counter = [0]

    def repl(match, display):
        counter[0] += 1
        ph = f"\x00MATH_{counter[0]}\x00"
        placeholders[ph] = (match.group(1), display)
        return ph

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: repl(m, True), text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", lambda m: repl(m, False), text)
    return text, placeholders





# ═══════════════════════════════════════════════════════════════════════
#  3. CODE PRE-RENDERING (Pygments)
# ═══════════════════════════════════════════════════════════════════════

_FORMATTER = HtmlFormatter(style="monokai", cssclass="code-block", nowrap=False)


def render_code(code: str, lang: str = "python") -> str:
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = PythonLexer() if lang == "python" else TextLexer()
    return highlight(code, lexer, _FORMATTER)


def get_pygments_css() -> tuple[str, str]:
    """Return (dark_css, light_css) for both themes."""
    dark = _FORMATTER.get_style_defs(".code-block")
    light_fmt = HtmlFormatter(style="default", cssclass="code-block", nowrap=False)
    light = light_fmt.get_style_defs(".code-block")
    return dark, light


# ═══════════════════════════════════════════════════════════════════════
#  4. KATEX CSS + FONT EMBEDDING
# ═══════════════════════════════════════════════════════════════════════

def get_inline_katex_css() -> str:
    """Read katex.min.css and embed fonts as base64 data URIs."""
    css_path = KATEX_DIR / "katex.min.css"
    with open(css_path) as f:
        css = f.read()

    fonts_dir = KATEX_DIR / "fonts"

    # Replace url(fonts/...) with base64 data URIs
    def font_repl(match):
        fname = match.group(1)
        fpath = fonts_dir / fname
        if not fpath.exists():
            return match.group(0)
        ext = fname.rsplit(".", 1)[-1]
        mime = {"woff2": "font/woff2", "woff": "font/woff", "ttf": "font/ttf"}.get(
            ext, "application/octet-stream"
        )
        with open(fpath, "rb") as ff:
            data = base64.b64encode(ff.read()).decode("ascii")
        return f'url(data:{mime};base64,{data})'

    css = re.sub(r"url\(fonts/([^)]+)\)", font_repl, css)
    return css


# ═══════════════════════════════════════════════════════════════════════
#  5. MARKDOWN → HTML (with pre-rendered math + code)
# ═══════════════════════════════════════════════════════════════════════

def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def extract_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Extract ```...``` blocks, replace with placeholders."""
    placeholders: dict[str, str] = {}
    counter = [0]
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

    def repl(m):
        counter[0] += 1
        lang = m.group(1) or "python"
        code = m.group(2)
        ph = f"\x00CODE_{counter[0]}\x00"
        placeholders[ph] = render_code(code, lang)
        return ph

    text = pattern.sub(repl, text)
    return text, placeholders


def protect_and_process(
    text: str,
    math_ph: dict[str, tuple[str, bool]],
    code_ph: dict[str, str],
) -> str:
    """Protect math/code from markdown processing, process markdown, then restore.

    Two-pass approach: replace protected content with markers → process markdown
    → restore markers with final HTML. This prevents pre-rendered HTML from being
    split across paragraph boundaries.
    """
    # Build protect/restore maps
    protect = {}  # marker → (type, data)
    counter = [0]

    def make_marker():
        counter[0] += 1
        return f"\x01PROTECT_{counter[0]}\x01"

    # Protect math placeholders
    for ph, (latex, display) in sorted(math_ph.items(), key=lambda x: -len(x[0])):
        marker = make_marker()
        protect[marker] = ("math", latex, display)
        text = text.replace(ph, marker)

    # Protect code placeholders
    for ph, rendered in sorted(code_ph.items(), key=lambda x: -len(x[0])):
        marker = make_marker()
        protect[marker] = ("code", rendered)
        text = text.replace(ph, marker)

    # Now process markdown (parsers see only markers, not real HTML)
    result = _process_markdown(text)

    # Restore markers with final HTML
    for marker, info in protect.items():
        if info[0] == "math":
            _, latex, display = info
            escaped_latex = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cls = "katex-display" if display else "katex-inline"
            span = f'<span class="{cls}">{escaped_latex}</span>'
            # Normalize whitespace before lookup (notebook LaTeX often has newlines)
            normalized = " ".join(latex.split())
            reading = EQ_READINGS.get(normalized, "")
            if reading:
                escaped_reading = reading.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                reading_html = (
                    '<button type="button" class="eq-reading-toggle" aria-expanded="false"'
                    ' title="How to read this" aria-label="Show how to read this equation">📖</button>'
                    f'<span class="eq-reading">{escaped_reading}</span>'
                )
                span += reading_html
            result = result.replace(marker, span)
        elif info[0] == "code":
            result = result.replace(marker, f'<div class="code-cell">{info[1]}</div>')

    return result


def _inline(text: str) -> str:
    """Apply inline markdown formatting: bold, italics, code, links."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _process_markdown(md: str) -> str:
    """Convert markdown to HTML. Markers for protected content are present but ignored."""

    # Split, keeping markers on their own lines
    lines = md.split("\n")
    out: list[str] = []
    in_table = False
    first_row = True
    list_tag: str | None = None  # None, "ol", or "ul"

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            is_sep = all(re.match(r"^[-: ]+$", c) for c in cells)
            if is_sep:
                i += 1
                continue
            if not in_table:
                close_list()
                out.append('<div class="table-wrapper"><table>')
                in_table = True
                first_row = True
            tag = "th" if first_row else "td"
            out.append(
                "<tr>"
                + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
                + "</tr>"
            )
            if first_row:
                first_row = False
            i += 1
            continue
        elif in_table:
            out.append("</table></div>")
            in_table = False

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.+)", stripped)
        if hm:
            close_list()
            level = len(hm.group(1))
            title = hm.group(2)
            if level == 2:
                slug = re.sub(r"^[^a-zA-Z]*", "", title)
                slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
                out.append(f'<h2 id="{slug}">{title}</h2>')
            elif level == 3:
                out.append(f'<h3 class="collapsible">{title}</h3>')
            else:
                out.append(f"<h{level}>{title}</h{level}>")
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        # Blockquotes (lens / warning callouts), merging consecutive "> " lines
        if stripped.startswith(">"):
            close_list()
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            body = " ".join(q for q in quote_lines if q)
            cls = "callout"
            low = body.lower()
            if "mathematician" in low:
                cls = "callout callout-math"
            elif "practitioner" in low:
                cls = "callout callout-warn"
            out.append(f'<blockquote class="{cls}"><p>{_inline(body)}</p></blockquote>')
            continue

        # Lists (ordered and unordered, with inline formatting)
        list_m = re.match(r"^(?:(\d+)\.|[-*])\s+(.+)", stripped)
        if list_m and not stripped.startswith("**"):
            tag = "ol" if list_m.group(1) else "ul"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline(list_m.group(2))}</li>")
            i += 1
            continue
        elif list_tag and not stripped:
            close_list()

        # Standalone protected block (code cell or display math) — no <p> wrapper
        if re.fullmatch(r"\x01PROTECT_\d+\x01", stripped):
            close_list()
            out.append(stripped)
            i += 1
            continue

        # Inline formatting
        if stripped:
            out.append(f"<p>{_inline(stripped)}</p>")
        else:
            out.append("")
        i += 1

    close_list()
    if in_table:
        out.append("</table></div>")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════
#  6. PEDAGOGICAL MARKUP
# ═══════════════════════════════════════════════════════════════════════

# Chapter metadata: objectives and key takeaways
CHAPTER_META = {
    0: {
        "title": "What A Language Model Computes",
        "difficulty": "Foundational",
        "time": "~15 min",
        "prereqs": ["Basic probability", "Matrix multiplication"],
        "objectives": [
            "Define an autoregressive language model as a parameterized family of conditional distributions",
            "Explain the chain-rule factorization of sequence probability",
            "Connect maximum likelihood training to negative log-likelihood minimization",
            "Identify the tensor shapes involved in next-token prediction",
        ],
        "takeaways": [
            "An autoregressive LM represents p(x_t | x_{&lt;t}) — a conditional distribution over a finite vocabulary",
            "Maximum likelihood training minimizes the average negative log-likelihood over observed token positions",
            "One text sequence yields many supervised prediction terms via label shifting",
            "The causal (lower-triangular) mask is what makes a decoder model autoregressive",
        ],
    },
    1: {
        "title": "Tensors, Autograd, And Probability",
        "difficulty": "Foundational",
        "time": "~25 min",
        "prereqs": ["Chapter 0", "Calculus (chain rule)", "Linear algebra basics"],
        "objectives": [
            "Understand how logits become probabilities via softmax",
            "Derive the cross-entropy gradient",
            "Explain autograd as mechanical chain-rule application",
            "Connect logistic regression to the one-logit warmup case",
        ],
        "takeaways": [
            "Softmax converts raw scores to a probability distribution via exp and normalization",
            "Cross-entropy loss = negative log-likelihood for the observed label",
            "Autograd computes gradients of a scalar loss w.r.t. all parameters via the chain rule",
            "The softmax Jacobian and cross-entropy gradient simplify elegantly: dL/dz = p - y",
        ],
    },
    2: {
        "title": "Tokenization: Text To Integer IDs",
        "difficulty": "Foundational",
        "time": "~20 min",
        "prereqs": ["Chapter 1"],
        "objectives": [
            "Understand how text becomes a finite sequence of integer token IDs",
            "Compare character-level vs subword (BPE) tokenization",
            "Explain why tokenization is not harmless preprocessing",
            "Connect tokenization to the model's vocabulary size and embedding table",
        ],
        "takeaways": [
            "Tokenization maps text to integer sequences over a finite vocabulary",
            "Character tokenizers are simple but produce long sequences; BPE is shorter but learned",
            "The vocabulary size V determines the embedding table rows and the logit dimension",
            "Different tokenizers create different conditional prediction problems for the same text",
        ],
    },
    3: {
        "title": "Embeddings And Next-Token Language Modeling",
        "difficulty": "Intermediate",
        "time": "~25 min",
        "prereqs": ["Chapter 2", "Dot products", "Matrix multiplication"],
        "objectives": [
            "Understand embeddings as a learned linear map from one-hot vectors to a continuous space",
            "Explain position embeddings and the residual stream",
            "Derive label shifting for next-token prediction",
            "Connect embedding geometry to model behavior",
        ],
        "takeaways": [
            "An embedding table is a learned matrix W_e ∈ R^{V×C} mapping token IDs to vectors",
            "Position embeddings add temporal information to the residual stream",
            "Label shifting: targets at position t come from input at position t+1",
            "Embedding geometry is learned from the training objective, not from pre-existing semantics",
        ],
    },
    4: {
        "title": "Attention From Raw Tensors",
        "difficulty": "Intermediate",
        "time": "~30 min",
        "prereqs": ["Chapter 3", "Matrix multiplication", "Softmax"],
        "objectives": [
            "Derive scaled dot-product attention from first principles",
            "Understand the role of Q, K, V projections",
            "Explain the causal mask and why it is needed",
            "Analyze the T×T attention bottleneck",
        ],
        "takeaways": [
            "Attention computes weighted averages of value vectors, where weights come from Q·Kᵀ softmax",
            "Scaling by 1/√d_k prevents softmax saturation as dimension grows",
            "The causal mask sets upper-triangular scores to -∞, preventing attention to future tokens",
            "Attention cost is O(T²·d) — the quadratic bottleneck motivates sparse/linear alternatives",
        ],
    },
    5: {
        "title": "The Transformer Block",
        "difficulty": "Intermediate",
        "time": "~25 min",
        "prereqs": ["Chapter 4", "Layer normalization", "MLP / feedforward layers"],
        "objectives": [
            "Compose attention, LayerNorm, MLP, and residual connections into a transformer block",
            "Understand pre-norm vs post-norm architecture",
            "Calculate parameter counts for each component",
            "Explain why residual connections are essential for training deep stacks",
        ],
        "takeaways": [
            "A transformer block = LayerNorm → Attention → Residual → LayerNorm → MLP → Residual",
            "Pre-norm (norm before sublayer) is more stable for deep stacks",
            "The MLP typically expands to 4× the model dimension then projects back",
            "Weight tying between embedding and LM head reduces parameters and improves training",
        ],
    },
    6: {
        "title": "Training: Loss, Batches, Optimizer, And Overfitting",
        "difficulty": "Intermediate",
        "time": "~25 min",
        "prereqs": ["Chapter 5", "Gradient descent", "Basic optimization"],
        "objectives": [
            "Understand the training loop structure for a language model",
            "Explain empirical risk minimization as the training objective",
            "Diagnose overfitting via training vs validation loss gap",
            "Connect perplexity to average negative log-likelihood",
        ],
        "takeaways": [
            "Training minimizes average per-token cross-entropy across batches",
            "AdamW is the standard optimizer; it combines momentum with decoupled weight decay",
            "Overfitting shows as train loss ↓ while val loss plateaus or rises",
            "Perplexity = exp(average NLL) — it's the effective vocabulary size the model is choosing among",
        ],
    },
    7: {
        "title": "Generation And Evaluation",
        "difficulty": "Intermediate",
        "time": "~20 min",
        "prereqs": ["Chapter 6"],
        "objectives": [
            "Understand autoregressive generation as iterative sampling",
            "Compare greedy, temperature, top-k, and top-p decoding strategies",
            "Explain prefill vs decode phases",
            "Evaluate generation quality beyond qualitative samples",
        ],
        "takeaways": [
            "Generation samples from p(x_t | x_{&lt;t}) one token at a time",
            "Temperature scales logits before softmax: high T flattens, low T sharpens",
            "Top-k and top-p (nucleus) truncate the distribution to reduce low-probability noise",
            "Qualitative samples can mislead — use perplexity and benchmark metrics for evaluation",
        ],
    },
    8: {
        "title": "Toy Supervised Fine-Tuning",
        "difficulty": "Intermediate",
        "time": "~20 min",
        "prereqs": ["Chapter 7"],
        "objectives": [
            "Understand how loss masking converts some positions into supervised targets",
            "Explain the instruction-response format for SFT",
            "Connect weighted empirical risk to loss masking",
            "Distinguish pretraining from supervised fine-tuning objectives",
        ],
        "takeaways": [
            "SFT masks prompt tokens so only response tokens contribute to loss",
            "The instruction format creates a structured (prompt, response) pair",
            "Loss masking = weighting each position's loss by 0 or 1",
            "SFT changes the data distribution, not the model architecture",
        ],
    },
    9: {
        "title": "Translation To Hugging Face And PyTorch Abstractions",
        "difficulty": "Intermediate",
        "time": "~15 min",
        "prereqs": ["Chapters 5-6"],
        "objectives": [
            "Map from-scratch components to Hugging Face abstractions",
            "Understand encoder, decoder, and encoder-decoder model families",
            "Identify which implementation details transfer to library code",
        ],
        "takeaways": [
            "TinyGPT maps to GPT2LMHeadModel in the Hugging Face ecosystem",
            "Datasets, tokenizers, and Trainer abstractions wrap the from-scratch pipeline",
            "Encoder models use bidirectional attention; decoder models use causal masking",
        ],
    },
    10: {
        "title": "Quantization: Lower Precision Representations",
        "difficulty": "Advanced",
        "time": "~25 min",
        "prereqs": ["Chapter 9", "Integer arithmetic"],
        "objectives": [
            "Understand quantization as mapping real tensors to a finite integer grid",
            "Compute scale and zero-point for symmetric and asymmetric quantization",
            "Analyze error sources: rounding, clipping, per-tensor vs per-channel",
            "Calculate KV-cache memory savings from quantization",
        ],
        "takeaways": [
            "Quantization maps [r_min, r_max] → [q_min, q_max] via scale and zero-point",
            "Per-channel quantization is more accurate than per-tensor but requires more metadata",
            "INT8 KV-cache reduces memory by ~4× with minimal quality loss",
            "The main error sources are rounding error and clipping error",
        ],
    },
    11: {
        "title": "Modern LLM Practice",
        "difficulty": "Advanced",
        "time": "~25 min",
        "prereqs": ["Chapter 8"],
        "objectives": [
            "Map the post-training pipeline: pretraining → SFT → preference optimization",
            "Understand RAG and tool use as problem decomposition",
            "Explain prefill/decode scheduling for inference serving",
            "Identify what lives inside vs outside the base LM",
        ],
        "takeaways": [
            "Modern LLM training has 3 stages: pretraining, SFT, and preference optimization (DPO/RLHF)",
            "RAG, tools, and memory are external to the base LM — they wrap it",
            "Inference serving separates prefill (parallel) from decode (sequential) phases",
            "Evaluation must go beyond perplexity — use benchmarks, human eval, and safety checks",
        ],
    },
    12: {
        "title": "Beyond Transformers And World Models",
        "difficulty": "Advanced",
        "time": "~30 min",
        "prereqs": ["Chapter 4", "Chapter 5"],
        "objectives": [
            "Identify the quadratic attention bottleneck and its alternatives",
            "Compare sparse, linear, and grouped-query attention variants",
            "Understand state-space models (Mamba) as recurrent alternatives",
            "Connect JEPA and world-model ideas to representation learning",
        ],
        "takeaways": [
            "Attention's O(T²) cost motivates sparse, linear, and recurrent alternatives",
            "GQA/MQA reduce KV-cache memory by sharing K/V heads",
            "State-space models (Mamba) achieve O(T) inference via selective recurrence",
            "World models learn representations that support prediction, not just token generation",
        ],
    },
    13: {
        "title": "How To Study From Here",
        "difficulty": "Foundational",
        "time": "~10 min",
        "prereqs": ["All chapters"],
        "objectives": [
            "Build a systematic study checklist for new LLM concepts",
            "Identify resources for deeper study",
            "Connect the toy implementation to real-world LLM systems",
        ],
        "takeaways": [
            "For each new concept: state the math, attach shapes, derive the formula, find the code path",
            "The toy implementation is a legible version of real production systems",
            "Stay grounded in the math — it's the anchor that prevents hand-waving",
        ],
    },
}


def make_objectives_banner(ch_num: int) -> str:
    meta = CHAPTER_META.get(ch_num)
    if not meta:
        return ""
    items = "".join(f"<li>{o}</li>" for o in meta["objectives"])
    prereqs = ", ".join(meta["prereqs"])
    return f"""
<div class="chapter-meta">
  <div class="meta-tags">
    <span class="tag tag-{meta['difficulty'].lower()}">{meta['difficulty']}</span>
    <span class="tag tag-time">{meta['time']}</span>
    <span class="tag tag-prereq">Prereqs: {prereqs}</span>
  </div>
  <div class="objectives">
    <h4>Chapter Objectives</h4>
    <ul>{items}</ul>
  </div>
</div>"""


def make_takeaways_box(ch_num: int) -> str:
    meta = CHAPTER_META.get(ch_num)
    if not meta:
        return ""
    items = "".join(f"<li>{t}</li>" for t in meta["takeaways"])
    return f"""
<div class="takeaways">
  <h4>🔑 Key Takeaways</h4>
  <ul>{items}</ul>
</div>"""


def make_quiz_block(ch_num: int) -> str:
    """Generate interactive self-check quizzes from chapter content."""
    # These are derived from the "Self-checks" sections in the notebook
    quizzes = QUIZZES.get(ch_num, [])
    if not quizzes:
        return ""
    blocks = []
    for i, q in enumerate(quizzes):
        blocks.append(f"""
<div class="quiz-block" data-quiz="{ch_num}-{i}">
  <div class="quiz-question">Q{i+1}. {q['question']}</div>
  <button class="quiz-reveal-btn" onclick="revealQuiz('{ch_num}-{i}')">Show Answer</button>
  <div class="quiz-answer" id="quiz-{ch_num}-{i}">{q['answer']}</div>
</div>""")
    return f'<div class="quiz-section"><h4>📝 Self-Check Questions</h4>{"".join(blocks)}</div>'


# Quiz data extracted from notebook 99 self-check sections
QUIZZES = {
    0: [
        {"question": "If logits have shape [8, 16, 100], how many multiclass classification examples are used by the cross-entropy call after flattening?", "answer": "8 × 16 = 128 examples. Each (batch, position) pair becomes one multiclass classification problem over 100 classes."},
        {"question": "Why is the sequence log-likelihood a sum but the sequence probability a product?", "answer": "The chain rule of probability gives p(x₁:T) = ∏ p(xₜ | x<ₜ). Taking logs converts the product into a sum: log p(x₁:T) = Σ log p(xₜ | x<ₜ)."},
        {"question": "Where in TinyGPT.forward does the model become a distribution over vocabulary items rather than a hidden representation?", "answer": "At the lm_head projection: logits = self.lm_head(x). The output shape [B, T, V] represents unnormalized log-probabilities over the vocabulary."},
    ],
    1: [
        {"question": "Why must the training objective be scalar before calling backward()?", "answer": "Autograd computes ∂L/∂θ for a scalar L. A non-scalar output cannot be differentiated without an additional vector-Jacobian product to collapse it to a scalar."},
        {"question": "If logits has shape [B, T, V], why does cross-entropy treat the final axis differently from the first two?", "answer": "Cross-entropy interprets the last axis as class logits (the V vocabulary items) and the first two as independent examples. PyTorch's F.cross_entropy expects (N, C, ...) layout."},
        {"question": "What is the simplified gradient of cross-entropy loss w.r.t. logits?", "answer": "∂L/∂z = softmax(z) - y_onehot = p - y. The softmax Jacobian and cross-entropy combine to give this clean expression."},
    ],
    4: [
        {"question": "Why scale Q·Kᵀ by 1/√d_k before softmax?", "answer": "Without scaling, large d_k makes Q·Kᵀ values large, pushing softmax into saturated regions with vanishing gradients. Scaling by 1/√d_k keeps the variance of scores near 1."},
        {"question": "What is the attention cost for sequence length T and head dimension d?", "answer": "O(T²·d) for the Q·Kᵀ matrix multiplication, and O(T²) for the softmax. The T² term is the quadratic bottleneck."},
        {"question": "What does the causal mask do to attention weights?", "answer": "It sets scores for positions j > i to -∞, so after softmax those positions get exactly 0 attention weight. Token i can only attend to tokens 0..i."},
    ],
    5: [
        {"question": "In pre-norm, where is the LayerNorm applied relative to the sublayer?", "answer": "Pre-norm applies LayerNorm before the sublayer: x = x + sublayer(LN(x)). This is more stable for deep stacks than post-norm."},
        {"question": "If n_embd=384, what is the MLP hidden dimension?", "answer": "4 × 384 = 1536. The standard feedforward expansion factor is 4×."},
        {"question": "Why tie weights between the embedding and LM head?", "answer": "Weight tying reduces parameters (V×C shared instead of 2×V×C), improves training signal, and is theoretically motivated: the inverse of the embedding map should produce logits."},
    ],
    6: [
        {"question": "What is the relationship between perplexity and average negative log-likelihood?", "answer": "Perplexity = exp(average NLL). It represents the effective number of tokens the model is equally uncertain between."},
        {"question": "How does overfitting manifest in the loss curves?", "answer": "Training loss continues to decrease while validation loss plateaus or increases. The gap between train and val loss grows."},
    ],
    7: [
        {"question": "What happens to the probability distribution as temperature → 0?", "answer": "As T→0, softmax(z/T) approaches argmax(z) — the distribution becomes one-hot on the highest logit (greedy decoding)."},
        {"question": "What is the difference between top-k and top-p (nucleus) sampling?", "answer": "Top-k keeps the k highest-probability tokens. Top-p keeps the smallest set of tokens whose cumulative probability ≥ p. Top-p adapts to the distribution shape; top-k is fixed."},
    ],
    10: [
        {"question": "If a tensor has range [-2.0, 3.0] and we quantize to INT8 [-128, 127], what are the scale and zero-point?", "answer": "Scale = (3.0 - (-2.0)) / (127 - (-128)) = 5.0/255 ≈ 0.0196. Zero-point = -128 - round(-2.0/0.0196) = -128 + 102 = -26."},
        {"question": "Why is per-channel quantization more accurate than per-tensor?", "answer": "Different channels may have different value ranges. Per-tensor uses one scale for all, causing large errors for channels with small ranges. Per-channel gives each channel its own scale."},
    ],
}


# ═══════════════════════════════════════════════════════════════════════
#  7. VISUALIZATION FRAGMENTS
# ═══════════════════════════════════════════════════════════════════════

SOFTMAX_VIZ = VIZ_CONTAINERS["softmax"]

ATTENTION_VIZ = VIZ_CONTAINERS["attention"]

TRANSFORMER_VIZ = VIZ_CONTAINERS["transformer"]

TRAINING_VIZ = VIZ_CONTAINERS["training"]

GENERATION_VIZ = VIZ_CONTAINERS["generation"]
EMBEDDING_VIZ = VIZ_CONTAINERS["embedding"]
QUANT_VIZ = VIZ_CONTAINERS["quant"]
KV_VIZ = VIZ_CONTAINERS["kv"]


# ═══════════════════════════════════════════════════════════════════════
#  8. CSS (all inline)
# ═══════════════════════════════════════════════════════════════════════

def build_css() -> str:
    katex_css = get_inline_katex_css()
    dark_pg, light_pg = get_pygments_css()
    # Prefix each rule with theme selector (CSS doesn't nest blocks)
    def prefix_rules(css_block: str, selector: str) -> str:
        lines = []
        for line in css_block.strip().split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("/*"):
                # Handle multi-selector rules
                parts = stripped.split("{", 1)
                if len(parts) == 2:
                    selectors = [s.strip() for s in parts[0].split(",")]
                    prefixed = ", ".join(f"{selector} {s}" for s in selectors)
                    lines.append(f"{prefixed} {{" + parts[1])
                else:
                    lines.append(line)
            else:
                lines.append(line)
        return "\n".join(lines)
    pygments_css = (
        "/* Pygments — dark (Monokai) */\n"
        + prefix_rules(dark_pg, '[data-theme="dark"]')
        + "\n/* Pygments — light (Default) */\n"
        + prefix_rules(light_pg, '[data-theme="light"]')
    )
    return f"""
{katex_css}
{pygments_css}

/* ═══ Reset & Variables ═══ */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #0d1117; --bg-panel: #161b22; --bg-elevated: #1c2333;
  --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
  --accent: #f0883e; --accent-glow: rgba(240,136,62,0.27);
  --accent-2: #58a6ff; --accent-3: #3fb950; --accent-4: #d2a8ff;
  --border: #30363d; --border-faint: #21262d;
  --code-bg: #0d1117; --link: #58a6ff;
  --radius: 8px; --radius-sm: 4px;
  --font-mono: 'SF Mono','Cascadia Code','Fira Code','JetBrains Mono',monospace;
  --font-body: 'Charter','Georgia',serif;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --sidebar-w: 280px; --transition: 160ms ease;
  --font-scale: 1;
}}

[data-theme="light"] {{
  --bg: #ffffff; --bg-panel: #f6f8fa; --bg-elevated: #eaeef2;
  --text: #1f2328; --text-dim: #636c76; --text-bright: #0a0c10;
  --border: #d0d7de; --border-faint: #e1e4e8;
  --code-bg: #f6f8fa; --link: #0969da;
  --accent: #bc4c00; --accent-glow: rgba(188,76,0,0.15);
  --accent-2: #0969da; --accent-3: #1a7f37; --accent-4: #8250df;
}}

html {{ scroll-behavior: smooth; font-size: calc(17px * var(--font-scale)); }}
body {{ background: var(--bg); color: var(--text); font-family: var(--font-body); line-height: 1.7; display: flex; min-height: 100vh; }}

/* ═══ Progress Bar ═══ */
#progress-bar {{
  position: fixed; top: 0; left: 0; height: 3px; width: 0%;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  z-index: 100; transition: width 100ms ease;
}}

/* ═══ Top Bar ═══ */
#topbar {{
  position: fixed; top: 0; left: var(--sidebar-w); right: 0; height: 0;
  z-index: 50; pointer-events: none;
}}
#topbar > * {{ pointer-events: auto; }}

#toolbar {{
  position: fixed; top: 12px; right: 20px; z-index: 50;
  display: flex; gap: 8px; align-items: center;
  font-family: var(--font-sans); font-size: 0.8rem;
}}

#toolbar button {{
  background: var(--bg-panel); border: 1px solid var(--border);
  color: var(--text-dim); border-radius: var(--radius-sm);
  padding: 6px 10px; cursor: pointer; transition: all var(--transition);
  font-size: 0.85rem;
}}
#toolbar button:hover {{ color: var(--text-bright); border-color: var(--accent); }}

/* ═══ Sidebar ═══ */
#sidebar {{
  position: fixed; top: 0; left: 0; width: var(--sidebar-w); height: 100vh;
  background: var(--bg-panel); border-right: 1px solid var(--border);
  overflow-y: auto; padding: 24px 20px; z-index: 10;
  font-family: var(--font-sans); font-size: 0.84rem;
  transition: transform var(--transition);
}}
#sidebar::-webkit-scrollbar {{ width: 6px; }}
#sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

#sidebar h3 {{ color: var(--accent); font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 16px; font-weight: 600; }}
#sidebar ul {{ list-style: none; }}
#sidebar li {{ margin-bottom: 1px; }}
#sidebar a {{ display: flex; align-items: baseline; gap: 8px; color: var(--text-dim); text-decoration: none; padding: 5px 10px; border-radius: var(--radius-sm); transition: all var(--transition); }}
#sidebar a:hover {{ color: var(--text-bright); background: var(--bg-elevated); }}
#sidebar a.active {{ color: var(--accent); background: var(--accent-glow); }}
#sidebar .ch-num {{ font-family: var(--font-mono); font-size: 0.68rem; color: var(--accent); min-width: 1.4em; text-align: right; opacity: 0.7; }}

/* ═══ Main ═══ */
#main {{ margin-left: var(--sidebar-w); flex: 1; max-width: 960px; padding: 60px 64px 120px; min-width: 0; }}

h1 {{ font-family: var(--font-sans); font-size: 2.4rem; font-weight: 700; color: var(--text-bright); line-height: 1.2; margin-bottom: 0.6em; letter-spacing: -0.02em; }}
h2 {{ font-family: var(--font-sans); font-size: 1.55rem; font-weight: 600; color: var(--text-bright); margin-top: 2.4em; margin-bottom: 0.6em; padding-top: 0.5em; border-top: 1px solid var(--border-faint); }}
h3 {{ font-family: var(--font-sans); font-size: 1.15rem; font-weight: 600; color: var(--accent); margin-top: 1.8em; margin-bottom: 0.5em; cursor: pointer; user-select: none; }}
h3.collapsible::before {{ content: '▸ '; font-size: 0.8em; transition: transform var(--transition); display: inline-block; width: 1em; }}
h3.collapsible.expanded::before {{ transform: rotate(90deg); }}
h3.collapsible:hover {{ color: var(--text-bright); }}
h4 {{ font-family: var(--font-sans); font-size: 1rem; font-weight: 600; color: var(--text-bright); margin-top: 1.2em; margin-bottom: 0.3em; }}
p {{ margin-bottom: 0.9em; }}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ font-family: var(--font-mono); font-size: 0.88em; background: var(--code-bg); padding: 0.1em 0.35em; border-radius: var(--radius-sm); border: 1px solid var(--border-faint); }}
strong {{ color: var(--text-bright); font-weight: 600; }}
hr {{ border: none; border-top: 1px solid var(--border-faint); margin: 2em 0; }}

/* ═══ Code Cells ═══ */
.code-cell {{ margin: 1.2em 0; border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); background: var(--code-bg); position: relative; }}
.code-cell pre {{ margin: 0; padding: 20px 24px; overflow-x: auto; font-size: 0.82rem; line-height: 1.55; white-space: pre-wrap; word-break: break-all; overflow-wrap: break-word; }}
.code-cell .code-block {{ background: none !important; }}
[data-theme="light"] .code-block, [data-theme="light"] .code-block span {{ color: #24292e !important; }}
.copy-btn {{ position: absolute; top: 8px; right: 8px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-dim); border-radius: var(--radius-sm); padding: 4px 8px; font-size: 0.72rem; cursor: pointer; opacity: 0; transition: opacity var(--transition); font-family: var(--font-sans); }}
.code-cell:hover .copy-btn {{ opacity: 1; }}
.copy-btn:hover {{ color: var(--accent); border-color: var(--accent); }}
.copy-btn.copied {{ color: var(--accent-3); border-color: var(--accent-3); }}

/* ═══ Tables ═══ */
.table-wrapper {{ overflow-x: visible; margin: 1em 0; border: 1px solid var(--border); border-radius: var(--radius); }}
table {{ width: 100%; border-collapse: collapse; font-family: var(--font-sans); font-size: 0.82rem; table-layout: auto; }}
th, td {{ padding: 8px 14px; text-align: left; border-bottom: 1px solid var(--border-faint); word-break: break-word; overflow-wrap: break-word; }}
th {{ background: var(--bg-panel); color: var(--text-bright); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
tr:last-child td {{ border-bottom: none; }}

/* ═══ KaTeX ═══ */
.katex {{ font-size: 1.08em !important; }}
.katex-display {{ display: block; margin: 1.2em 0; overflow-x: auto; overflow-y: hidden; padding: 0.2em 0; }}
.katex-display > .katex {{ font-size: 1.15em !important; }}

/* ═══ Chapter Meta ═══ */
.chapter-meta {{ margin: 1em 0 2em; padding: 16px 20px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: var(--radius); }}
.meta-tags {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
.tag {{ font-family: var(--font-sans); font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 12px; border: 1px solid var(--border); }}
.tag-foundational {{ color: var(--accent-3); border-color: var(--accent-3); background: rgba(63,185,80,0.1); }}
.tag-intermediate {{ color: var(--accent-2); border-color: var(--accent-2); background: rgba(88,166,255,0.1); }}
.tag-advanced {{ color: var(--accent); border-color: var(--accent); background: var(--accent-glow); }}
.tag-time {{ color: var(--text-dim); }}
.tag-prereq {{ color: var(--text-dim); }}
.objectives h4 {{ color: var(--accent); margin-bottom: 8px; }}
.objectives ul {{ padding-left: 20px; }}
.objectives li {{ margin-bottom: 4px; font-size: 0.9rem; }}

/* ═══ Takeaways ═══ */
.takeaways {{ margin: 2em 0; padding: 16px 20px; background: rgba(63,185,80,0.08); border: 1px solid var(--accent-3); border-radius: var(--radius); }}
.takeaways h4 {{ color: var(--accent-3); margin-bottom: 8px; }}
.takeaways ul {{ padding-left: 20px; }}
.takeaways li {{ margin-bottom: 6px; font-size: 0.9rem; }}

/* ═══ Quiz ═══ */
.quiz-section {{ margin: 2em 0; }}
.quiz-block {{ margin: 1em 0; padding: 12px 16px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: var(--radius); }}
.quiz-question {{ font-weight: 600; color: var(--text-bright); margin-bottom: 8px; }}
.quiz-reveal-btn {{ font-family: var(--font-sans); font-size: 0.78rem; font-weight: 600; padding: 5px 14px; border-radius: var(--radius-sm); border: 1px solid var(--accent-2); background: rgba(88,166,255,0.1); color: var(--accent-2); cursor: pointer; transition: all var(--transition); }}
.quiz-reveal-btn:hover {{ background: var(--accent-2); color: var(--bg); }}
.quiz-answer {{ display: none; margin-top: 10px; padding: 10px 14px; background: var(--bg-elevated); border-left: 3px solid var(--accent-2); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; font-size: 0.88rem; }}
.quiz-answer.visible {{ display: block; }}

/* ═══ Collapsible sections ═══ */
.collapsible-content {{ overflow: hidden; transition: max-height 300ms ease; }}

/* ═══ Visualizations ═══ */
.viz-container {{ margin: 2em 0; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-panel); overflow: hidden; }}
.viz-header {{ display: flex; align-items: center; gap: 10px; padding: 12px 20px; background: var(--bg-elevated); border-bottom: 1px solid var(--border); font-family: var(--font-sans); font-size: 0.82rem; }}
.viz-icon {{ font-size: 1.1rem; }}
.viz-title {{ color: var(--text-bright); font-weight: 600; flex: 1; }}
.viz-hint {{ color: var(--text-dim); font-size: 0.74rem; font-style: italic; }}
.viz-body {{ padding: 16px 20px; }}
.viz-body canvas {{ display: block; margin: 0 auto; border-radius: var(--radius-sm); background: var(--bg); max-width: 100%; }}
.viz-controls {{ display: flex; align-items: center; gap: 14px; margin-top: 14px; font-family: var(--font-sans); font-size: 0.82rem; flex-wrap: wrap; }}
.viz-controls label {{ color: var(--text-dim); display: flex; align-items: center; gap: 8px; }}
.viz-controls label span {{ color: var(--accent); font-family: var(--font-mono); font-weight: 600; min-width: 2em; }}
.viz-controls input[type="range"] {{ -webkit-appearance: none; width: 180px; height: 4px; background: var(--border); border-radius: 2px; outline: none; }}
.viz-controls input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 16px; height: 16px; background: var(--accent); border-radius: 50%; cursor: pointer; border: 2px solid var(--bg); box-shadow: 0 0 8px var(--accent-glow); }}
.viz-controls button {{ font-family: var(--font-sans); font-size: 0.8rem; font-weight: 600; padding: 8px 18px; border-radius: var(--radius-sm); border: 1px solid var(--accent); background: var(--accent-glow); color: var(--accent); cursor: pointer; transition: all var(--transition); }}
.viz-controls button:hover {{ background: var(--accent); color: var(--bg); }}
.gen-output {{ font-family: var(--font-mono); font-size: 0.9rem; color: var(--accent-3); padding: 4px 12px; background: var(--code-bg); border: 1px solid var(--border); border-radius: var(--radius-sm); margin-left: auto; }}
.attention-panels {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
.attn-panel {{ text-align: center; }}
.panel-label {{ font-family: var(--font-sans); font-size: 0.73rem; color: var(--text-dim); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
.attn-info {{ margin-top: 16px; padding: 12px 16px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 0.77rem; color: var(--accent-2); text-align: center; min-height: 2.6em; display: flex; align-items: center; justify-content: center; }}
.attn-placeholder {{ color: var(--text-dim); }}

/* ═══ Equation Readings ═══ */
.eq-reading-toggle {{
  display: inline-block; cursor: pointer; font-size: 0.75em;
  margin-left: 4px; opacity: 0.4; transition: opacity var(--transition);
  user-select: none; vertical-align: middle; border: 0; padding: 0;
  color: inherit; background: transparent; font-family: inherit;
}}
.katex-display + .eq-reading-toggle {{
  display: block; margin-top: -8px; margin-bottom: 4px; font-size: 0.8em;
}}
.eq-reading-toggle:hover {{ opacity: 0.7; }}
.eq-reading-toggle:focus-visible {{ opacity: 1; outline: 2px solid var(--accent-2); outline-offset: 2px; }}
.eq-reading-toggle.active {{ opacity: 0.8; }}
.eq-reading {{
  display: none; font-family: var(--font-sans); font-size: 0.82rem;
  line-height: 1.55; color: var(--text-dim);
  padding: 8px 14px; margin: 4px 0 10px;
  background: var(--bg-elevated); border-left: 3px solid var(--accent-4);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}}
.eq-reading.visible {{ display: block; }}
.sr-only {{
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
}}

/* ═══ Search ═══ */
#search-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 200; backdrop-filter: blur(4px); }}
#search-overlay.active {{ display: flex; align-items: flex-start; justify-content: center; padding-top: 120px; }}
#search-box {{ width: 90%; max-width: 600px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }}
#search-input {{ width: 100%; padding: 16px 20px; background: var(--bg-elevated); border: none; color: var(--text-bright); font-size: 1rem; font-family: var(--font-sans); outline: none; }}
#search-input::placeholder {{ color: var(--text-dim); }}
#search-results {{ max-height: 400px; overflow-y: auto; }}
.search-result {{ padding: 10px 20px; border-bottom: 1px solid var(--border-faint); cursor: pointer; font-family: var(--font-sans); font-size: 0.85rem; }}
.search-result:hover {{ background: var(--bg-elevated); }}
.search-result .sr-title {{ color: var(--text-bright); font-weight: 600; }}
.search-result .sr-context {{ color: var(--text-dim); font-size: 0.78rem; margin-top: 2px; }}

/* ═══ Viz control extensions (data-driven panels) ═══ */
.viz-readout {{ font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-dim); margin-left: 12px; }}
.viz-controls select {{ background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 3px 6px; font-family: var(--font-mono); font-size: 0.8rem; }}
.viz-controls.viz-stack {{ flex-direction: column; align-items: flex-start; gap: 8px; }}
.viz-controls.viz-wrap {{ flex-wrap: wrap; row-gap: 8px; }}
.attention-panels {{ display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }}
.attn-side {{ display: flex; flex-direction: column; gap: 12px; min-width: 200px; }}

/* ═══ Back to Top ═══ */
#back-to-top {{ position: fixed; bottom: 24px; right: 24px; width: 40px; height: 40px; border-radius: 50%; background: var(--bg-panel); border: 1px solid var(--border); color: var(--accent); font-size: 1.2rem; cursor: pointer; opacity: 0; transition: opacity var(--transition); z-index: 30; display: flex; align-items: center; justify-content: center; }}
#back-to-top.visible {{ opacity: 1; }}
#back-to-top:hover {{ border-color: var(--accent); background: var(--accent-glow); }}

/* ═══ Mobile ═══ */
#nav-toggle {{ display: none; position: fixed; top: 16px; left: 16px; z-index: 20; width: 40px; height: 40px; border-radius: var(--radius-sm); background: var(--bg-panel); border: 1px solid var(--border); color: var(--text-bright); font-size: 1.2rem; cursor: pointer; align-items: center; justify-content: center; }}
@media (max-width: 1024px) {{
  #sidebar {{ transform: translateX(-100%); width: 260px; }}
  #sidebar.open {{ transform: translateX(0); }}
  #main {{ margin-left: 0; padding: 40px 20px 80px; }}
  h1 {{ font-size: 1.8rem; }}
  code {{ overflow-wrap: anywhere; word-break: break-word; }}
  .table-wrapper {{ overflow-x: auto; }}
  .attention-panels {{ flex-direction: column; align-items: center; }}
  #nav-toggle {{ display: flex; }}
  #toolbar {{ right: 12px; }}
}}
"""


# ═══════════════════════════════════════════════════════════════════════
#  9. JAVASCRIPT (all inline)
# ═══════════════════════════════════════════════════════════════════════

def build_js(ch_starts_json: str) -> str:
    return f"""
// ═══ Theme ═══
function getTheme() {{ return localStorage.getItem('theme') || 'dark'; }}
function setTheme(t) {{
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('theme', t);
  document.getElementById('theme-icon').textContent = t === 'dark' ? '☀️' : '🌙';
}}
setTheme(getTheme());

// ═══ Font Size ═══
function getFontScale() {{ return parseFloat(localStorage.getItem('fontScale') || '1'); }}
function setFontScale(s) {{
  s = Math.max(0.8, Math.min(1.4, s));
  document.documentElement.style.setProperty('--font-scale', s);
  localStorage.setItem('fontScale', s);
}}
setFontScale(getFontScale());

// ═══ Sidebar ═══
function toggleSidebar() {{ document.getElementById('sidebar').classList.toggle('open'); }}

const chStarts = {ch_starts_json};
const navLinks = document.querySelectorAll('#sidebar a');
function updateActiveNav() {{
  const scrollY = window.scrollY + 80;
  let active = null;
  for (const [slug] of chStarts) {{
    const el = document.getElementById(slug);
    if (el && el.offsetTop <= scrollY) active = slug;
  }}
  navLinks.forEach(a => {{
    const href = a.getAttribute('href');
    a.classList.toggle('active', href === '#' + active);
  }});
}}

// ═══ Progress Bar ═══
function updateProgress() {{
  const h = document.documentElement.scrollHeight - window.innerHeight;
  const p = h > 0 ? (window.scrollY / h) * 100 : 0;
  document.getElementById('progress-bar').style.width = p + '%';
}}

// ═══ Back to Top ═══
function updateBackToTop() {{
  document.getElementById('back-to-top').classList.toggle('visible', window.scrollY > 400);
}}

window.addEventListener('scroll', () => {{ updateActiveNav(); updateProgress(); updateBackToTop(); }}, {{ passive: true }});

// ═══ Collapsible h3 sections ═══
function initCollapsibles() {{
  document.querySelectorAll('h3.collapsible').forEach((h3, index) => {{
    let content = [];
    let next = h3.nextElementSibling;
    while (next && !next.matches('h2, h3.collapsible')) {{
      content.push(next);
      next = next.nextElementSibling;
    }}
    const wrapper = document.createElement('div');
    wrapper.className = 'collapsible-content';
    wrapper.id = 'collapsible-content-' + (index + 1);
    wrapper.style.maxHeight = 'none';
    h3.parentNode.insertBefore(wrapper, content[0] || h3.nextSibling);
    content.forEach(el => wrapper.appendChild(el));

    h3.tabIndex = 0;
    h3.classList.add('expanded');
    h3.setAttribute('role', 'button');
    h3.setAttribute('aria-controls', wrapper.id);
    h3.setAttribute('aria-expanded', 'true');

    const toggle = () => {{
      h3.classList.toggle('expanded');
      if (h3.classList.contains('expanded')) {{
        wrapper.style.maxHeight = wrapper.scrollHeight + 'px';
      }} else {{
        wrapper.style.maxHeight = '0px';
      }}
      h3.setAttribute('aria-expanded', String(h3.classList.contains('expanded')));
    }};
    h3.addEventListener('click', toggle);
    h3.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); toggle(); }}
    }});
  }});
}}

function initEquationReadings() {{
  document.querySelectorAll('.eq-reading-toggle').forEach((button, index) => {{
    const reading = button.nextElementSibling;
    if (!reading) return;
    reading.id = 'eq-reading-' + (index + 1);
    button.setAttribute('aria-controls', reading.id);
    button.addEventListener('click', () => {{
      reading.classList.toggle('visible');
      button.classList.toggle('active');
      const expanded = reading.classList.contains('visible');
      button.setAttribute('aria-expanded', String(expanded));
      button.setAttribute('aria-label', expanded
        ? 'Hide how to read this equation'
        : 'Show how to read this equation');
    }});
  }});
}}

// ═══ Quiz Reveal ═══
function revealQuiz(id) {{
  const ans = document.getElementById('quiz-' + id);
  const btn = document.querySelector('[data-quiz="' + id + '"] .quiz-reveal-btn');
  ans.classList.toggle('visible');
  btn.textContent = ans.classList.contains('visible') ? 'Hide Answer' : 'Show Answer';
}}

// ═══ Copy Code ═══
function initCopyButtons() {{
  document.querySelectorAll('.code-cell').forEach(cell => {{
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {{
      const code = cell.querySelector('pre')?.textContent || '';
      navigator.clipboard.writeText(code).then(() => {{
        btn.textContent = '✓ Copied';
        btn.classList.add('copied');
        setTimeout(() => {{ btn.textContent = 'Copy'; btn.classList.remove('copied'); }}, 2000);
      }});
    }});
    cell.appendChild(btn);
  }});
}}

// ═══ Search ═══
const searchableText = [];
function buildSearchIndex() {{
  const sections = document.querySelectorAll('main section');
  sections.forEach((sec, i) => {{
    const text = sec.textContent.toLowerCase();
    const headings = sec.querySelectorAll('h2, h3');
    headings.forEach(h => {{
      searchableText.push({{ title: h.textContent, text: text, element: sec, heading: h }});
    }});
  }});
}}

function performSearch(query) {{
  query = query.toLowerCase().trim();
  if (!query) return [];
  const results = [];
  for (const item of searchableText) {{
    const idx = item.text.indexOf(query);
    if (idx !== -1) {{
      const start = Math.max(0, idx - 60);
      const ctx = item.text.substring(start, idx + query.length + 60);
      results.push({{ title: item.title, context: '...' + ctx + '...', element: item.heading }});
      if (results.length >= 12) break;
    }}
  }}
  return results;
}}

function showSearch() {{
  document.getElementById('search-overlay').classList.add('active');
  setTimeout(() => document.getElementById('search-input').focus(), 50);
}}
function hideSearch() {{
  document.getElementById('search-overlay').classList.remove('active');
  document.getElementById('search-input').value = '';
  document.getElementById('search-results').innerHTML = '';
}}

document.addEventListener('keydown', (e) => {{
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {{ e.preventDefault(); showSearch(); }}
  if (e.key === 'Escape') hideSearch();
}});

// ═══ KaTeX Client-Side Rendering ═══
function renderAllMath() {{
  document.querySelectorAll('.katex-inline').forEach(el => {{
    const latex = el.textContent;
    try {{
      katex.render(latex, el, {{ throwOnError: false, displayMode: false, trust: true }});
    }} catch(e) {{ el.textContent = latex; }}
  }});
  document.querySelectorAll('.katex-display').forEach(el => {{
    const latex = el.textContent;
    try {{
      katex.render(latex, el, {{ throwOnError: false, displayMode: true, trust: true }});
    }} catch(e) {{ el.textContent = latex; }}
  }});
}}

// ═══ Init ═══
document.addEventListener('DOMContentLoaded', () => {{
  renderAllMath();
  initCollapsibles();
  initEquationReadings();
  initCopyButtons();
  buildSearchIndex();
  updateActiveNav();
  updateProgress();
  initSoftmaxViz();
  initAttentionViz();
  initTransformerViz();
  initTrainingViz();
  initGenerationViz();
  initEmbeddingViz();
  initQuantViz();
  initKVViz();

  // Theme toggle
  document.getElementById('theme-toggle').addEventListener('click', () => {{
    setTheme(getTheme() === 'dark' ? 'light' : 'dark');
  }});

  // Font size
  document.getElementById('font-dec').addEventListener('click', () => setFontScale(getFontScale() - 0.1));
  document.getElementById('font-inc').addEventListener('click', () => setFontScale(getFontScale() + 0.1));

  // Back to top
  document.getElementById('back-to-top').addEventListener('click', () => window.scrollTo({{ top: 0, behavior: 'smooth' }}));

  // Search
  document.getElementById('search-input').addEventListener('input', (e) => {{
    const results = performSearch(e.target.value);
    const container = document.getElementById('search-results');
    container.innerHTML = results.map(r =>
      '<div class="search-result" data-href="#' + (r.element?.id || '') + '"><div class="sr-title">' + r.title + '</div><div class="sr-context">' + r.context + '</div></div>'
    ).join('');
    container.querySelectorAll('.search-result').forEach(el => {{
      el.addEventListener('click', () => {{
        const href = el.getAttribute('data-href');
        if (href && href !== '#') {{
          document.getElementById(href.substring(1))?.scrollIntoView({{ behavior: 'smooth' }});
        }}
        hideSearch();
      }});
    }});
  }});

  document.getElementById('search-overlay').addEventListener('click', (e) => {{
    if (e.target.id === 'search-overlay') hideSearch();
  }});
}});

""" + VIZ_JS




# ═══════════════════════════════════════════════════════════════════════
#  10. HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

def build_html(
    cells: list[dict],
    css: str,
    js: str,
    katex_js_b64: str,
    viz_data_json: str = "null",
) -> str:
    chapter_starts = []
    sections_html = []
    chapter_num = -1
    prev_chapter_num = -1

    for i, cell in enumerate(cells):
        source = cell["source"]

        if cell["type"] == "markdown":
            # Detect chapter
            m = re.match(r"^## (\d+)\.\s", source.strip())
            if m:
                prev_chapter_num = chapter_num
                chapter_num = int(m.group(1))

                # Inject takeaways for previous chapter
                if prev_chapter_num >= 0:
                    sections_html.append(make_takeaways_box(prev_chapter_num))

                title = source.strip().split("\n")[0][3:].strip()
                slug = re.sub(r"^[^a-zA-Z]*", "", title)
                slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
                chapter_starts.append((i, len(sections_html), chapter_num, title, slug))

                # Inject objectives banner
                sections_html.append(make_objectives_banner(chapter_num))

            # Extract math and code, protect from markdown processing
            src_no_math, cell_math = extract_math(source)
            source_with_code, code_ph = extract_code_blocks(src_no_math)

            body = protect_and_process(
                source_with_code,
                cell_math,
                code_ph,
            )

            # Inject visualizations
            src_lower = source.lower()
            if chapter_num == 1 and "logistic regression" in src_lower:
                body += "\n" + SOFTMAX_VIZ
            elif chapter_num == 4 and "scaling and the causal mask" in src_lower:
                body += "\n" + ATTENTION_VIZ
            elif chapter_num == 5 and "the transformer block" in src_lower[:60]:
                body += "\n" + TRANSFORMER_VIZ
            elif chapter_num == 6 and "training" in src_lower[:25]:
                body += "\n" + TRAINING_VIZ
            elif chapter_num == 7 and "generation" in src_lower[:20]:
                body += "\n" + GENERATION_VIZ
            elif chapter_num == 3 and "static embeddings versus contextual" in src_lower:
                body += "\n" + EMBEDDING_VIZ
            elif chapter_num == 10 and "per-tensor versus per-channel quantization" in src_lower[:60]:
                body += "\n" + QUANT_VIZ
            elif chapter_num == 12 and "beyond transformers and world models" in src_lower[:60]:
                body += "\n" + KV_VIZ

            # Inject quiz at end of chapter (before next chapter)
            # We'll handle this at the next chapter boundary

            sections_html.append(f'<section class="cell-md">\n{body}\n</section>')

        elif cell["type"] == "code":
            rendered = render_code(source)
            sections_html.append(f'<section class="cell-code-wrap"><div class="code-cell">{rendered}</div></section>')

    # Inject takeaways for last chapter
    if chapter_num >= 0:
        sections_html.append(make_takeaways_box(chapter_num))

    # Build nav
    nav_items = []
    for _, sec_idx, cn, title, slug in chapter_starts:
        nav_items.append(
            f'<li><a href="#{slug}" data-section="{sec_idx}">'
            f'<span class="ch-num">{cn}</span>{title}</a></li>'
        )
    nav_html = "<ul>\n" + "\n".join(nav_items) + "\n</ul>"

    # Inject quizzes at chapter boundaries
    # We need to find where each chapter ends and insert the quiz
    final_sections = []
    ch_idx = 0
    for idx, sec in enumerate(sections_html):
        # Check if this is where a new chapter starts
        if ch_idx < len(chapter_starts) and idx == chapter_starts[ch_idx][1]:
            # Insert quiz for previous chapter before this chapter's objectives
            if ch_idx > 0:
                prev_ch = chapter_starts[ch_idx - 1][2]
                quiz = make_quiz_block(prev_ch)
                if quiz:
                    final_sections.append(quiz)
            ch_idx += 1
        final_sections.append(sec)

    # Add quiz for last chapter
    if chapter_starts:
        last_ch = chapter_starts[-1][2]
        quiz = make_quiz_block(last_ch)
        if quiz:
            final_sections.append(quiz)

    content_html = "\n".join(final_sections)
    ch_starts_json = json.dumps([[s[4], s[2]] for s in chapter_starts])
    js = js.replace("const chStarts = [];", f"const chStarts = {ch_starts_json};")

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mathematical Foundations of Language Model Systems — Interactive Textbook</title>
<style>{css}</style>
<script>{katex_js_b64}</script>
</head>
<body>

<div id="progress-bar"></div>

<div id="toolbar">
  <button id="search-btn" onclick="showSearch()" title="Search (⌘K)" aria-label="Search textbook">🔍</button>
  <button id="font-dec" title="Decrease font" aria-label="Decrease font size">A−</button>
  <button id="font-inc" title="Increase font" aria-label="Increase font size">A+</button>
  <button id="theme-toggle" title="Toggle theme" aria-label="Toggle color theme"><span id="theme-icon">☀️</span></button>
</div>

<button id="nav-toggle" onclick="toggleSidebar()" aria-label="Toggle navigation">☰</button>

<nav id="sidebar">
  <h3>Chapters</h3>
  {nav_html}
</nav>

<main id="main">
  <div id="byline">
    <span><a href="https://0xstrategies.com">Rafael Almeida</a> &middot; operations &times; AI systems</span>
    <span><em>Mathematical Foundations of Language Model Systems</em> &mdash; notebook, toy GPT, and render pipeline by the author &middot; <a href="https://github.com/zeroexstrat/lm-foundations">GitHub</a></span>
  </div>
  {content_html}
</main>

<style>
#byline {{
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: .07em;
  text-transform: uppercase; color: var(--text-dim);
  display: flex; flex-wrap: wrap; gap: 6px 28px; justify-content: space-between;
  padding: 14px 0 16px; border-bottom: 1px solid var(--border); margin-bottom: 30px;
}}
#byline a {{ color: var(--accent-2); text-decoration: none; }}
#byline a:hover {{ text-decoration: underline; }}
#byline em {{ font-style: normal; color: var(--text); }}
blockquote.callout {{
  margin: 22px 0; padding: 14px 18px;
  background: var(--bg-panel); border-left: 3px solid var(--accent-2);
  border-radius: var(--radius-sm);
}}
blockquote.callout p {{ margin: 0; }}
blockquote.callout-math {{ border-left-color: var(--accent-4); }}
blockquote.callout-warn {{ border-left-color: var(--accent); }}
</style>

<button id="back-to-top" title="Back to top" aria-label="Back to top">↑</button>

<div id="search-overlay" onclick="if(event.target.id==='search-overlay')hideSearch()">
  <div id="search-box">
    <input type="text" id="search-input" placeholder="Search the textbook... (Esc to close)">
    <div id="search-results"></div>
  </div>
</div>

<script>window.VIZ_DATA = {viz_data_json};</script>
<script>{js}</script>

</body>
</html>"""
    return template


def artifact_counts(html: str) -> dict[str, int]:
    """Count rendered textbook components without matching CSS or JavaScript text."""
    return {
        "chapters": html.count('<li><a href="#'),
        "math": (
            html.count('<span class="katex-inline">')
            + html.count('<span class="katex-display">')
        ),
        "code_blocks": html.count('<div class="code-cell">'),
        "visualizations": html.count('<div class="viz-container">'),
    }


def render_textbook() -> str:
    """Render the complete self-contained textbook without writing to disk."""
    cells = load_notebook()
    katex_js_text = base64.b64decode(get_katex_js_b64()).decode("utf-8")
    viz_path = ROOT / "output" / "viz_data.json"
    viz_data_json = viz_path.read_text(encoding="utf-8") if viz_path.exists() else "null"
    return build_html(cells, build_css(), build_js("[]"), katex_js_text, viz_data_json)


# ═══════════════════════════════════════════════════════════════════════
#  11. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Rendering notebook, KaTeX, styles, and interactions...")
    viz_path = ROOT / "output" / "viz_data.json"
    if viz_path.exists():
        print(f"Embedding real viz data ({viz_path.stat().st_size // 1024} KB) from viz_data.json...")
    else:
        print("NOTE: output/viz_data.json not found — interactives will show a")
        print("      'run scripts/viz_data.py' notice instead of live data.")

    html = render_textbook()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    size_kb = len(html) / 1024
    counts = artifact_counts(html)
    print("\n✅ Built self-contained interactive textbook")
    print(f"   → {OUTPUT_PATH}")
    print(f"   Size: {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
    print(f"   Chapters: {counts['chapters']}")
    print(f"   Math expressions: {counts['math']} (client-side KaTeX)")
    print(f"   Code blocks: {counts['code_blocks']} (Pygments)")
    print(f"   Interactive visualizations: {counts['visualizations']}")
    print("   External dependencies: 0")


if __name__ == "__main__":
    main()
