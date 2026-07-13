# Readings for the worked-example equations (auto-paired to normalized LaTeX keys).

EQ_READINGS_V3: dict[str, str] = {
    'p(\\texttt{c}) = 0.4, \\qquad p(\\texttt{a}\\mid\\texttt{c}) = 0.5, \\qquad p(\\texttt{t}\\mid\\texttt{ca}) = 0.6 .':
        'The three conditional probabilities assumed for the worked example: probability 0.4 for c to start, 0.5 for a after c, 0.6 for t after ca.',
    'P(\\texttt{cat}) = 0.4 \\times 0.5 \\times 0.6 = 0.12,':
        'The chain rule with digits: multiply the three conditionals to get the joint probability of the whole word — 0.12.',
    '-\\log P(\\texttt{cat}) = \\underbrace{0.9163}_{-\\ln 0.4} + \\underbrace{0.6931}_{-\\ln 0.5} + \\underbrace{0.5108}_{-\\ln 0.6} = 2.1203 \\text{ nats}.':
        'The negative log-likelihood term by term: each conditional contributes minus its natural log, and the three contributions sum to 2.1203 nats. One sequence, three supervised terms.',
    'z = (2,\\ 0,\\ -1,\\ 1).':
        'The example logit vector: four unconstrained real scores, one per vocabulary token (space, a, c, t).',
    'Z = 7.3891 + 1 + 0.3679 + 2.7183 = 11.4752, \\qquad A(z) = \\ln Z = 2.4402,':
        'The partition function computed by hand: sum the four exponentials to get Z, whose log is the log-partition function A of z.',
    '\\sigma(z) = (0.6439,\\ 0.0871,\\ 0.0321,\\ 0.2369) \\in \\mathring{\\Delta}^{3},':
        'The resulting probability vector after dividing each exponential by Z — four positive numbers summing to one, a point of the open simplex.',
    '\\ell(z, 3) = A(z) - z_3 = 2.4402 - 1 = 1.4402 \\text{ nats},':
        'Cross-entropy evaluated: the log-partition value minus the logit of the true class t gives 1.4402 nats.',
    '\\nabla_z \\ell = (0.6439,\\ 0.0871,\\ 0.0321,\\ \\mathbf{-0.7631}),':
        'The gradient vector p minus one-hot: three positive pushes down on wrong tokens, one negative push up on the true token, summing to zero exactly.',
    "\\operatorname{enc}: \\quad \\texttt{' '} \\mapsto 0, \\quad \\texttt{a} \\mapsto 1, \\quad \\texttt{c} \\mapsto 2, \\quad \\texttt{t} \\mapsto 3, \\qquad V = 4 .":
        "The running vocabulary: the tokenizer sorts the distinct characters of 'a cat' and numbers them — space is 0, a is 1, c is 2, t is 3.",
    '\\operatorname{enc}(\\texttt{"cat"}) = (2,\\ 1,\\ 3) \\in \\mathcal{V}^3,':
        'The word cat as an ID sequence: character by character, (2, 1, 3).',
    "W_E = \\begin{pmatrix} 0 & 0.1 \\\\ 0.5 & -0.5 \\\\ 1 & 0.5 \\\\ -0.5 & 1 \\end{pmatrix} \\begin{matrix} \\leftarrow \\texttt{' '} \\\\ \\leftarrow \\texttt{a} \\\\ \\leftarrow \\texttt{c} \\\\ \\leftarrow \\texttt{t} \\end{matrix} \\qquad W_P = \\begin{pmatrix} 0.1 & 0 \\\\ 0 & 0.1 \\\\ -0.1 & 0 \\end{pmatrix} \\begin{matrix} \\leftarrow t=1 \\\\ \\leftarrow t=2 \\\\ \\leftarrow t=3 \\end{matrix}":
        'The hand-chosen embedding table (4 rows, one per token, width 2) and position table (3 rows, one per position) used by every worked example.',
    'e_2 W_E = (0,\\ 0,\\ 1,\\ 0)\\,W_E = 0\\cdot(0, 0.1) + 0\\cdot(0.5, -0.5) + 1\\cdot(1, 0.5) + 0\\cdot(-0.5, 1) = (1,\\ 0.5).':
        'The lookup written as one-hot matrix multiplication: the one-hot vector for token c selects exactly row 2 of the table and zeroes out the rest.',
    'X = \\mathrm{In}(\\texttt{"cat"}) = \\begin{pmatrix} 1 & 0.5 \\\\ 0.5 & -0.5 \\\\ -0.5 & 1 \\end{pmatrix} + \\begin{pmatrix} 0.1 & 0 \\\\ 0 & 0.1 \\\\ -0.1 & 0 \\end{pmatrix} = \\begin{pmatrix} 1.1 & 0.5 \\\\ 0.5 & -0.4 \\\\ -0.6 & 1 \\end{pmatrix} \\in \\mathbb{R}^{3\\times 2}.':
        'The initial residual stream: token embedding rows for c, a, t plus the position rows, added entrywise — a 3-by-2 matrix, one row per position.',
    'Q = X = \\begin{pmatrix} 1.1 & 0.5 \\\\ 0.5 & -0.4 \\\\ -0.6 & 1 \\end{pmatrix}, \\qquad K = XW_K = \\begin{pmatrix} 0.5 & 1.1 \\\\ -0.4 & 0.5 \\\\ 1 & -0.6 \\end{pmatrix}, \\qquad V = X .':
        'Queries, keys, values for the worked head: Q and V are the residual stream itself; K is the stream with its two columns swapped.',
    'a_{3,1} = q_3 \\cdot k_1 = (-0.6)(0.5) + (1)(1.1) = -0.30 + 1.10 = 0.80 .':
        'One score entry fully expanded: the dot product of query 3 with key 1 is two multiplications and one addition, totalling 0.80.',
    '\\frac{A}{\\sqrt{2}} = \\begin{pmatrix} 0.7778 & -0.1344 & 0.5657 \\\\ -0.1344 & -0.2828 & 0.5233 \\\\ 0.5657 & 0.5233 & -0.8485 \\end{pmatrix} \\ \\xrightarrow{\\ +M\\ }\\ \\begin{pmatrix} 0.7778 & -\\infty & -\\infty \\\\ -0.1344 & -0.2828 & -\\infty \\\\ 0.5657 & 0.5233 & -0.8485 \\end{pmatrix}.':
        'The full score matrix divided by root 2 (the temperature), then the causal mask sets the upper triangle to minus infinity.',
    'W = \\begin{pmatrix} 1 & 0 & 0 \\\\ 0.5371 & 0.4629 & 0 \\\\ 0.4542 & 0.4354 & 0.1104 \\end{pmatrix} \\in \\mathcal{S}_3 \\quad (\\text{Definition 4.3: rows sum to } 1,\\ \\text{upper triangle } 0).':
        'The weight matrix after row-wise softmax: row 1 is forced to a vertex (nothing else to read), rows 2 and 3 are genuine distributions over their prefixes; upper triangle exactly zero.',
    'o_3 = 0.4542\\,v_1 + 0.4354\\,v_2 + 0.1104\\,v_3 = (0.6511,\\ 0.1634).':
        'The output at position 3 as an explicit convex combination: about 45% of value 1, 44% of value 2, 11% of itself.',
    '\\mu = 0.8, \\qquad u - \\mu\\mathbf{1} = (0.3,\\ -0.3), \\qquad \\sigma = 0.3, \\qquad N(u) = (1,\\ -1).':
        'Layer norm on one row: subtract the mean 0.8, divide by the standard deviation 0.3 — the result is exactly (1, -1), and it is (1, -1) for every input with first coordinate larger.',
    'm_1 = 0.9 \\cdot 0 + 0.1 \\cdot 0.5 = 0.05, \\qquad v_1 = 0.999 \\cdot 0 + 0.001 \\cdot 0.25 = 0.00025,':
        'The two exponential moving averages at step one: with zero initialization, m picks up one tenth of the gradient and v one thousandth of its square.',
    '\\hat m_1 = \\frac{0.05}{1 - 0.9} = 0.5, \\qquad \\hat v_1 = \\frac{0.00025}{1 - 0.999} = 0.25, \\qquad \\sqrt{\\hat v_1} = 0.5,':
        'The bias corrections at step one divide by exactly the EMA weights, recovering the raw gradient 0.5 and its square 0.25 — so the preconditioned ratio is exactly one.',
    '\\theta_1 = 1 - 10^{-3}\\left(\\frac{0.5}{0.5 + 10^{-8}} + 0.01 \\cdot 1\\right) = 1 - 10^{-3}(1 + 0.01) = 0.99899 .':
        'The completed update: learning rate times (unit preconditioned step plus decoupled decay), moving theta from 1 to 0.99899 — the same result for any gradient magnitude.',
    'p(\\,\\cdot \\mid S) = \\left(\\frac{0.6439}{0.8808},\\ \\frac{0.2369}{0.8808}\\right) = (0.7311,\\ 0.2689) = \\sigma\\big((2, 1)\\big).':
        'Top-2 truncation renormalized: dividing the two surviving probabilities by their sum 0.8808 gives exactly the softmax of the two surviving logits — conditioning commutes with the Gibbs map.',
    's = \\frac{x_{\\max} - x_{\\min}}{q_{\\max} - q_{\\min}} = \\frac{2}{7} = 0.2857, \\qquad z = \\operatorname{round}\\!\\Big(0 - \\frac{-1}{2/7}\\Big) = \\operatorname{round}(3.5) = 4,':
        'The 3-bit calibration: scale is the real range 2 divided by the 7 integer gaps; the zero point rounds -x_min over s (3.5 rounds to 4, half to even).',
}
