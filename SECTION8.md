# Section 8: Adam optimizer + training loop

Adam optimizer with bias correction, linear learning rate decay, and the main training loop.

## Adam hyperparameters

| Scheme | Python | Value | Purpose |
|---|---|---|---|
| `learning-rate` | `learning_rate` | 0.01 | base learning rate |
| `beta1` | `beta1` | 0.85 | first moment decay |
| `beta2` | `beta2` | 0.99 | second moment decay |
| `eps-adam` | `eps_adam` | 1e-8 | numerical stability |

## Adam buffers

`adam-m` and `adam-v` are vectors of length `num-params`, storing the first and second moment estimates. Using vectors for O(1) indexed access during the optimizer update, which iterates over all 4192 parameters every step.

`params-vec` is `params` converted to a vector so the optimizer loop can index into it with `vector-ref` instead of `list-ref`.

## `tokenize`

Converts a document string to a vector of token IDs: `[BOS, char-id-1, char-id-2, ..., BOS]`. Uses `hash-table-ref char->id ch` for O(1) encoding. Returns a vector (not a list) since the training loop indexes into it by position.

## Training loop

A `do` loop over `num-steps` (1000) steps. Each step:

1. **Pick document**: `(vector-ref docs (modulo step (vector-length docs)))` — O(1) since `docs` is a vector.
2. **Tokenize**: convert to a vector of token IDs with BOS on both ends.
3. **Forward pass**: for each position in the sequence, run `gpt` to get logits, apply `softmax`, compute negative log-likelihood of the target token. Accumulate per-position losses.
4. **Compute mean loss**: `(v* (vsum losses) (/ 1.0 n))` — average over positions.
5. **Backward pass**: `(backward loss)` propagates gradients through the entire computation graph.
6. **Adam update**: for each parameter, update first/second moment estimates, apply bias correction, and update the parameter's data. Linear learning rate decay: `lr_t = lr * (1 - step/num_steps)`.
7. **Reset gradients**: set each parameter's grad to 0 for the next step.

The inner position loop uses a named `let` (`pos-loop`) that accumulates losses. The Adam update uses a `do` loop with vector indexing.

## Decisions

- **`params-vec` for O(1) optimizer access**. `params` is a list (from `append-map` in section 6). The optimizer iterates by index, so we convert to a vector once. An alternative would be to iterate `params` as a list with a counter, but vector indexing is cleaner and matches the moment buffer access pattern.
- **`tokenize` returns a vector**. The training loop accesses `tokens[pos_id]` and `tokens[pos_id + 1]` — random access that would be O(n) with a list. Vector gives O(1).
- **`list-ref probs target-id`**. The softmax output `probs` is a list of length `vocab-size` (27). Accessing the target token's probability uses `list-ref`, which is O(n) where n=27. For phase 2 with 600+ tokens this should become a vector — noted in OPTIMIZATION.md if needed.
- **`ensure-value` in `v/`**. The Python's `Value.__truediv__` handles `Value / int` via `__pow__`. Our `v/` calls `v** b -1` where `b` might be a plain number (e.g., `(length x)` in `rmsnorm`). Adding `ensure-value` to `v/` matches how `v+` and `v*` handle mixed arguments.
- **`(/ 1.0 n)` instead of `(/ 1 n)`**. Using `1.0` ensures floating-point division. With `(/ 1 n)`, Scheme would produce an exact rational `1/n`, which would propagate through the autograd graph as an exact number — correct but potentially slower.
