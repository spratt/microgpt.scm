# Known optimization opportunities

Tracked non-performant implementation details that are acceptable in phase 1 but should be addressed for phase 2.

## `weighted-choice`: two passes instead of one (SECTION2)

`weighted-choice` computes `(fold + 0 weights)` twice — once for `total` (which is then unused) and once inside the `threshold` calculation. In phase 2 with 600+ tokens, this doubles the work unnecessarily.

Fix: compute `total` once and reuse it.

```scheme
(let* ((total (fold + 0 weights))
       (threshold (* (random-real) total)))
  ...)
```

## `docs`: list instead of vector (SECTION3)

`docs` is a list, accessed during training with `list-ref` by `step % len(docs)`. This is O(n) where n is up to 32K documents. Python lists are array-backed (O(1)), so this is a regression.

Fix: `(define docs (list->vector (shuffle (read-lines "input.txt"))))` and use `vector-ref` / `vector-length` in the training loop.
