# Design: microgpt.scm

A faithful, idiomatic translation of Karpathy's microgpt.py from Python to R7RS-small Scheme. Same algorithm, Scheme idioms instead of Python OOP.

Performance decisions — vectors over lists for indexed access, hash tables for lookups — are made with scale in mind (e.g. 600+ token vocabulary vs the current 27).

## Libraries

R7RS-small:
- `(scheme base)` — core language
- `(scheme inexact)` — `exp`, `log`, `sqrt`, `acos`
- `(scheme file)` — `call-with-input-file`, `read-line`
- `(scheme write)` — `display`
- `(scheme char)` — `char-whitespace?`

SRFIs (supported by both Chibi Scheme and CHICKEN):
- `(srfi 1)` — `fold`, `iota`, `take`, `append-map`, `drop-while`, `delete-duplicates`
- `(srfi 27)` — `random-integer`, `random-real`
- `(srfi 69)` — `make-hash-table`, `hash-table-ref`, `hash-table-set!`, `hash-table-exists?`, `hash-table-keys`
- `(srfi 132)` — `list-sort`

Rejected: SRFI-13 (string utilities, not available in Chibi), SRFI-125 (hash tables, not available as a CHICKEN egg). Guile was dropped as a target because it lacks SRFI-132.

## Design decisions

### 1. Value (autograd node) → `define-record-type`

Python's `Value` class becomes a Scheme record with four fields: `data` and `grad` (mutable — updated by the optimizer and backward pass), `children` and `local-grads` (immutable — set at node creation).

Each arithmetic operation (`v+`, `v*`, `v**`, `vlog`, `vexp`, `vrelu`) creates a new Value recording its inputs (children) and local partial derivatives (local-grads). The backward pass walks the resulting computation graph in reverse, applying the chain rule.

`ensure-value` auto-wraps plain numbers so `v+` and `v*` accept mixed Value/number arguments, matching Python's `__radd__` and `__rmul__`.

The backward pass uses a SRFI-69 hash table keyed by `eq?` identity as a visited set for topological sort, matching Python's `set()` which uses object identity. No `visited` flag is needed on the record.

### 2. PRNG — SRFI-27 + helpers

SRFI-27 provides uniform random numbers. Three helpers built on top:

- **Box-Muller transform**: converts two uniform samples to one Gaussian sample. Used for weight initialization (std=0.08).
- **Fisher-Yates shuffle**: in-place vector shuffle. Used once at startup to randomize training data order.
- **Cumulative distribution sampling**: walks a list of weights, returns the index where the running sum exceeds a random threshold. Used during inference to sample from the model's predicted distribution.

Outputs differ from Python (different PRNG algorithm) but the training algorithm is identical.

### 3. State dict → SRFI-69 hash table

Python's `state_dict` dictionary maps string keys to weight matrices. We mirror this with a SRFI-69 hash table using `string=?` and `string-hash`. The `gpt` function accesses weights via `(hash-table-ref state-dict "wte")` etc., staying close to the Python.

The `params` flat list is built by iterating `hash-table-keys` and flattening each matrix with `apply append`. Iteration order is unspecified, but the optimizer treats all parameters identically.

### 4. Matrices → vectors of lists

Weight matrices have two access patterns:
- **Indexed row lookup**: `wte[token_id]` — needs O(1) random access.
- **Row-by-row iteration**: `linear(x, w)` — needs sequential traversal.

We use **vectors of lists**: the outer dimension is a vector (O(1) row access via `vector-ref`), each row is a list of Value nodes (natural for `map`/`fold` in dot products). The hidden state `x` flows through the network as a list. `linear` converts the matrix to a list with `vector->list` for `map`.

### 5. KV cache → vectors of lists with mutation

Each layer has its own growing list of past key/value vectors. The outer structure is a vector of length `n-layer` (O(1) layer indexing), each slot holding a list that grows via `append`. The cache is re-created fresh for each training document and each inference sample.

### 6. Tokenizer → hash table + vector

Encoding (char → token id) uses a SRFI-69 hash table for O(1) lookup, replacing Python's `uchars.index(ch)` which is O(n). Decoding (token id → char) uses `uchars` stored as a vector for O(1) access via `vector-ref`.

### 7. Dataset → vector

`docs` is stored as a vector, not a list. The training loop accesses documents by `step % len(docs)` — with 32K documents, `list-ref` would be O(n) per step. `list->vector` is called once at startup.

### 8. Attention head output — position-first iteration

The naive translation of the attention weighted sum iterates by dimension, using `list-ref` into each value vector — O(head_dim^2) per position. We restructure to iterate by position, scaling each value vector by its attention weight and summing element-wise via `fold`. This is O(head_dim) per position, which matters at scale where head_dim is typically ~64.

### 9. Adam buffers → vectors

The optimizer's first-moment (`adam-m`) and second-moment (`adam-v`) buffers are vectors for O(1) indexed access. `params` is also converted to a vector (`params-vec`) so the optimizer loop can index into it.

### 10. `string-trim` — hand-rolled

SRFI-13 is not available in Chibi Scheme. `string-trim` is implemented using `drop-while` from SRFI-1 on both ends of the character list.

### 11. Default hash functions

SRFI-69 hash function signatures differ between implementations (Chibi expects two arguments, CHICKEN expects one). Using `(make-hash-table)` with defaults avoids this incompatibility. For `state-dict`, we explicitly pass `string=?` and `string-hash` since string keys are known.

## File structure

Single file `microgpt.scm` containing (in order):
1. Imports
2. PRNG helpers (Box-Muller, shuffle, weighted-choice)
3. Dataset loading (read-lines, shuffle, convert to vector)
4. Tokenizer (uchars vector, char->id hash table, BOS, vocab-size)
5. Autograd engine (Value record, operations, backward)
6. Model parameters (hyperparameters, matrix constructor, state-dict, params)
7. Model architecture (linear, softmax, rmsnorm, gpt)
8. Adam optimizer + training loop
9. Inference + sampling

## Verification

- Chibi Scheme: `chibi-scheme microgpt.scm`
- CHICKEN: `csc -R r7rs microgpt.scm -o microgpt && ./microgpt`
- Training loss decreases from ~3.3 (random baseline, ln(27)) to ~2.0 over 1000 steps
- Inference produces plausible name-like strings (e.g., "senija", "jayli", "axilia")
