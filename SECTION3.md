# Section 3: Dataset loading

Reads `input.txt` (32,033 names), trims whitespace, filters empty lines, shuffles.

## `string-trim`

Hand-rolled replacement for SRFI-13's `string-trim-both`. Converts to a list of chars, drops leading whitespace with `drop-while`, reverses, drops trailing whitespace, reverses back. Uses `char-whitespace?` from `(scheme char)`.

## `read-lines`

Uses `call-with-input-file` and `read-line` (both R7RS-small) to read all lines from a file. Each line is trimmed; empty lines are discarded. Returns a list of strings.

## Top-level

```scheme
(define docs (shuffle (read-lines "input.txt")))
```

Reads and shuffles in one expression. The Python downloads `input.txt` if missing; we skip that since R7RS-small has no HTTP client. The file must be present.

## Decisions

- **No auto-download**. Python's `urllib.request.urlretrieve` has no R7RS equivalent. The file is already in the repo.
- **`string-trim` uses `drop-while`** from SRFI-1, which we already import. This avoids index arithmetic and is idiomatic Scheme.
- **`read-lines` returns a list**, matching how `docs` is used throughout (indexed by `step % len(docs)` via `list-ref`). In phase 2 with a larger dataset, this could become a vector for O(1) access.
