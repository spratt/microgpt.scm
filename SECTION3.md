# Section 3: Dataset loading

Reads `input.txt` (32,033 names), trims whitespace, filters empty lines, shuffles.

## `string-trim`

Hand-rolled replacement for SRFI-13's `string-trim-both`. Converts to a list of chars, drops leading whitespace with `drop-while`, reverses, drops trailing whitespace, reverses back. Uses `char-whitespace?` from `(scheme char)`.

## `read-lines`

Uses `call-with-input-file` and `read-line` (both R7RS-small) to read all lines from a file. Each line is trimmed; empty lines are discarded. Returns a list of strings.

## Top-level

```scheme
(define docs (list->vector (shuffle (read-lines "input.txt"))))
```

Reads, shuffles, and converts to a vector in one expression. The Python downloads `input.txt` if missing; we skip that since R7RS-small has no HTTP client. The file must be present.

## Decisions

- **No auto-download**. Python's `urllib.request.urlretrieve` has no R7RS equivalent. The file is already in the repo.
- **`string-trim` uses `drop-while`** from SRFI-1, which we already import. This avoids index arithmetic and is idiomatic Scheme.
- **`docs` is a vector**, not a list. `read-lines` returns a list, which `shuffle` randomizes, then `list->vector` converts to a vector for O(1) indexed access. The training loop will use `vector-ref` by `step % (vector-length docs)`. With 32K documents, `list-ref` would be O(n) per step — a significant regression from Python's array-backed lists. The tokenizer's `uchars` computation converts back to a list with `vector->list` for `append-map`, but this runs once at startup.
