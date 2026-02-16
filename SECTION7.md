# Section 7: Model architecture

Four functions implementing the GPT forward pass: `linear`, `softmax`, `rmsnorm`, and `gpt`.

## `linear` — matrix-vector multiply

Iterates over rows of weight matrix `w` (a vector of lists), computing a dot product of each row with input `x`. Uses `vector->list` to convert the matrix for `map`, and `vsum`/`v*` for the dot product. Returns a list of Values.

## `softmax` — numerically stable softmax

Subtracts the max logit value (as a plain number via `value-data`) before exponentiating, preventing overflow. The subtraction uses `v+` with a negated plain number (auto-wrapped by `ensure-value`). Returns a list of Values representing a probability distribution.

## `rmsnorm` — root mean square normalization

Computes `scale = (mean(x^2) + 1e-5)^(-0.5)` and multiplies each element by it. All operations go through the autograd graph so gradients flow through the normalization.

## `list-slice` — list slicing helper

`(list-slice lst start len)` returns `len` elements starting at index `start`. Built from SRFI-1's `take` and R7RS's `list-tail`. Used to split q/k/v vectors into per-head slices in the attention computation.

## `gpt` — full forward pass

Takes a token ID, position ID, and mutable key/value cache vectors. Returns a list of logits (one per vocab token).

The structure follows the Python exactly:
1. Look up token and position embeddings (O(1) via `vector-ref` on the weight matrices)
2. Sum embeddings and apply rmsnorm
3. For each layer: multi-head attention with residual connection, then MLP with residual connection
4. Final linear projection to vocab-size logits

The layer loop uses a named `let` (`layer-loop`) that threads `x` through iterations, since each layer transforms `x` and passes it to the next.

### Multi-head attention

The head loop (`head-loop`) accumulates head outputs into a flat list via `append`. For each head:
1. Slice q, k, v into per-head chunks using `list-slice`
2. Compute attention logits: `dot(q_h, k_t) / sqrt(head_dim)` for each cached key
3. Apply softmax to get attention weights
4. Compute weighted sum of cached value vectors
5. Append head output to the accumulator

The attention value computation iterates by position, scaling each value vector by its attention weight and summing element-wise via `fold`. This avoids `list-ref` entirely — O(positions * head_dim) instead of O(positions * head_dim^2).

## Decisions

- **Named `let` for the layer loop**. A `do` loop can't thread `x` through iterations cleanly — the body would need `set!` on a variable that `let*` shadows. A named `let` (`layer-loop`) passes `x` as a loop parameter, which is both correct and idiomatic.
- **`-inf.0` as initial max value**. `fold max -inf.0 ...` handles any list of real numbers. R7RS guarantees `-inf.0` exists when `(scheme inexact)` is imported.
- **Position-first iteration for head output**. A naive translation would iterate by dimension `j` and use `list-ref vt j` into each value vector — O(head_dim^2) per position due to `list-ref` being O(j). Instead, we `fold` over positions, scaling each value vector by its attention weight and accumulating element-wise with `map`. This is O(head_dim) per position and critical for phase 2 where head_dim will be ~64.
- **`append` for head accumulation**. Each head produces `head-dim` elements. We `append` them into a growing list across heads. With `n-head = 4` and `head-dim = 4`, this produces a 16-element list matching `n-embd`.
- **Scale as plain number**. `(/ 1.0 (sqrt head-dim))` is computed once per head as a plain number, then multiplied into the autograd graph via `v*` (which auto-wraps it). This avoids creating an unnecessary Value node for a constant.
