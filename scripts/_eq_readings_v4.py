# Readings for the bridge/motivation-pass equations.

EQ_READINGS_V4: dict[str, str] = {
    '\\texttt{the cat sat . the cat ran . the dog sat .}':
        'The three-sentence word-level corpus of the worked example: twelve word instances over a six-word vocabulary.',
    '\\hat P(\\texttt{cat} \\mid \\texttt{the}) = \\frac{2}{3}, \\qquad \\hat P(\\texttt{dog} \\mid \\texttt{the}) = \\frac{1}{3},':
        "Bigram maximum-likelihood estimates from visible counts: 'the cat' occurred 2 of the 3 times 'the' occurred, 'the dog' once.",
    '\\hat P(\\texttt{ran} \\mid \\texttt{dog}) = \\frac{0}{1} = 0,':
        "The zero-probability catastrophe: 'ran' never followed 'dog' in the corpus, so count-and-divide assigns it exactly zero — and one zero factor kills the whole sequence.",
    '\\hat P_{+1}(w \\mid \\texttt{dog}) = \\frac{\\operatorname{count}(\\texttt{dog}, w) + 1}{\\operatorname{count}(\\texttt{dog}) + V} \\quad\\Longrightarrow\\quad \\hat P_{+1}(\\texttt{ran} \\mid \\texttt{dog}) = \\frac{0 + 1}{1 + 6} = \\frac{1}{7} \\approx 0.143,':
        "Add-1 smoothing: pretend every continuation after 'dog' was seen once more — add 1 to each count, add the vocabulary size V to the denominator. The unseen event now gets 1/7.",
    'x = (2,\\ 1,\\ 3,\\ 0), \\qquad y = (1,\\ 3,\\ 0,\\ 1),':
        "The shifted pair for 'cat a': input IDs x are the first four tokens, target IDs y are the same window moved one step left — four supervised examples in one pass.",
    '\\text{bytes} = 2 \\cdot L \\cdot H \\cdot D \\cdot T \\cdot B \\cdot 2 = 2 \\cdot 2 \\cdot 4 \\cdot 8 \\cdot 16 \\cdot 1 \\cdot 2 = 4096 = 4\\,\\text{KB}.':
        'The KV-cache formula evaluated on the toy configuration: 2 (keys and values) times 2 layers times 4 heads times 8 dimensions times 16 positions times batch 1 times 2 bytes = 4096 bytes.',
}
