# Section 4: Tokenizer

Maps between characters and integer token IDs. Output: `vocab size: 27` (26 lowercase letters + BOS).

## `uchars` — unique sorted characters (vector)

Collects all characters from all docs with `append-map string->list`, deduplicates with `delete-duplicates` (SRFI-1), sorts with `list-sort` (SRFI-132) using `char<?`, then converts to a vector with `list->vector`. Matches Python's `sorted(set(''.join(docs)))`. Stored as a vector for O(1) decoding (token ID → char via `vector-ref`).

## `char->id` — hash table for encoding

A SRFI-69 hash table mapping each character to its token index (e.g., `#\a` → 0, `#\b` → 1, ..., `#\z` → 25). Gives O(1) lookup, replacing Python's `uchars.index(ch)` which is O(n).

## `BOS` and `vocab-size`

`BOS` is the token ID for the Beginning of Sequence special token, equal to `(vector-length uchars)` = 26. `vocab-size` is `BOS + 1` = 27.

## Decisions

- **Default hash function for `make-hash-table`**. We tried providing `char->integer` as a custom hash function, but Chibi's SRFI-69 expects hash functions with a different signature (two arguments: key and bound), causing a segfault. Using `(make-hash-table)` with defaults works on both Chibi and CHICKEN since the default `equal?`/`hash` handles chars correctly.
- **`uchars` is a vector, not a list**. Decoding (token ID → char) uses `vector-ref` for O(1) access. The sort/dedup pipeline produces a list, which is converted to a vector with `list->vector` at the end. This matters for phase 2 where the vocab will be 600+ tokens — O(n) `list-ref` would be called for every generated token during inference. The `char->id` hash table construction iterates the vector with a `do` loop instead of `fold` over a list.
