# Section 2: PRNG helpers

Three helpers built on top of SRFI-27's `random-real` and `random-integer`.

## `gauss` — Gaussian random numbers

Uses the Box-Muller transform to convert two uniform samples into a Gaussian sample:

```
sqrt(-2 * log(u1)) * cos(2 * pi * u2)
```

Replaces Python's `random.gauss(mean, std)`. Used for weight initialization — each of the ~5000 parameters is drawn from a normal distribution with std=0.08.

## `shuffle` — Fisher-Yates shuffle

Converts a list to a vector, shuffles in-place by walking from the end and swapping each element with a random earlier element, then converts back to a list.

Replaces Python's `random.shuffle(docs)`. Used once at startup to randomize the order of training documents.

## `weighted-choice` — cumulative distribution sampling

Walks a list of weights, accumulating a running sum. Returns the index where the cumulative sum exceeds a uniform random threshold.

Replaces Python's `random.choices(range(vocab_size), weights=...)`. Used during inference to sample the next token from the model's predicted probability distribution.

## Decisions

- **`shuffle` operates on lists** via list-to-vector-to-list conversion. Fisher-Yates requires O(1) indexed access for swaps, which vectors provide. The input dataset (`docs`) is a list, so we convert at the boundary.
- **`weighted-choice` computes `total` but doesn't use it** — the threshold is computed from a second `fold`. This is a minor inefficiency (two passes instead of one) but keeps the code clear. The weight list is only 27 elements (vocab_size), so it's negligible.
