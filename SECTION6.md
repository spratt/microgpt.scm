# Section 6: Model parameters

Hyperparameters, weight initialization, state dict, and the flat params list.

## Hyperparameters

Five constants matching the Python, using Scheme's kebab-case convention:

| Scheme | Python | Value | Purpose |
|---|---|---|---|
| `n-layer` | `n_layer` | 1 | transformer depth |
| `n-embd` | `n_embd` | 16 | embedding dimension |
| `block-size` | `block_size` | 16 | max context length |
| `n-head` | `n_head` | 4 | attention heads |
| `head-dim` | `head_dim` | 4 | per-head dimension (n-embd / n-head) |

## `matrix` — weight matrix constructor

Creates a vector of `nout` rows, each row a list of `nin` Values initialized from a Gaussian with mean=0, std=0.08. Uses `iota` from SRFI-1 to generate a dummy list of length `nin`, then `map` to create the Values.

The Python version takes `std` as a parameter (default 0.08). Since all call sites use the default, we hardcode it.

## `state-dict` — SRFI-69 hash table

A string-keyed hash table mapping weight names to matrices. Built imperatively with `hash-table-set!` inside a `let` block. The `do` loop over `n-layer` builds per-layer key names using `string-append`.

Contents (for `n-layer=1`):
- `"wte"`: 27×16 (token embeddings)
- `"wpe"`: 16×16 (position embeddings)
- `"lm_head"`: 27×16 (output projection)
- `"layer0.attn_wq"`: 16×16 (attention query)
- `"layer0.attn_wk"`: 16×16 (attention key)
- `"layer0.attn_wv"`: 16×16 (attention value)
- `"layer0.attn_wo"`: 16×16 (attention output)
- `"layer0.mlp_fc1"`: 64×16 (MLP expansion)
- `"layer0.mlp_fc2"`: 16×64 (MLP contraction)

Total: 4192 parameters.

## `params` — flat parameter list

Flattens all weight matrices into a single list of Value nodes for the Adam optimizer. Uses `hash-table-keys` to iterate over all entries, `vector->list` to convert each matrix's rows from a vector to a list, and `apply append` to concatenate all rows.

The iteration order over `hash-table-keys` is unspecified, but this doesn't matter — the optimizer treats all parameters identically. What matters is that `params` contains every Value node exactly once.

## Decisions

- **Hardcoded std=0.08**. The Python `matrix` lambda has `std=0.08` as a default, and all call sites use the default. No need for a parameter.
- **`string=?` and `string-hash` for the hash table**. Unlike the `char->id` table (section 4) where we used defaults, here we explicitly provide `string=?` and `string-hash` since the keys are strings. This is clearer and avoids relying on `equal?` dispatch.
- **`apply append` instead of `append-map identity`**. To flatten a list of lists, `(apply append lst)` is more direct than `(append-map (lambda (x) x) lst)`. Used to flatten each matrix's rows into a single list.
- **`hash-table-keys` for iteration**. SRFI-69 provides `hash-table-keys` which returns a list of all keys. We use this with `append-map` to build the flat params list. An alternative would be `hash-table-walk`, but `append-map` over keys is more functional.
