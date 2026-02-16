# Optimization tracking

## Implemented

### `weighted-choice`: single pass (SECTION2)

`weighted-choice` previously computed `(fold + 0 weights)` twice — once for `total` and once inline in `threshold`. Fixed to compute `total` once via `let*` and reuse it.

### `docs`: vector instead of list (SECTION3)

`docs` was a list, accessed during training with `list-ref` by `step % len(docs)` — O(n) where n is up to 32K. Fixed to `(define docs (list->vector (shuffle (read-lines "input.txt"))))` with `vector-ref` / `vector-length` for O(1) access.

### Attention head output: O(positions * head_dim^2) → O(positions * head_dim) (SECTION7)

The head output computation originally iterated by dimension `j`, indexing into each value vector with `list-ref vt j` — O(j) per access, O(head_dim^2) total per position. Fixed to iterate by position instead, scaling each value vector by its attention weight and summing element-wise via `fold`. Only sequential list traversal, no `list-ref`. Critical for phase 2 where head_dim will be ~64.

## Not yet implemented

(none)
