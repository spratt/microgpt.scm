# Section 5: Value record type + autograd operations + backward

The core autograd engine. Translates Python's `Value` class into a Scheme record type and free functions.

## `<value>` record type

Four fields: `data` (mutable — updated by optimizer), `grad` (mutable — accumulated during backward, reset to 0 after), `children` (immutable — input nodes), `local-grads` (immutable — chain rule factors). Matches the Python `__slots__`.

## `val` and `ensure-value`

`val` wraps a plain number as a leaf Value node (no children, no gradients). `ensure-value` auto-wraps plain numbers so that `v+` and `v*` can accept mixed Value/number arguments, matching Python's `__radd__` and `__rmul__` behavior.

## Autograd operations

Each operation creates a new Value node recording its inputs (children) and local partial derivatives (local-grads):

| Scheme | Python | Local gradients |
|---|---|---|
| `v+` | `__add__` | (1, 1) |
| `v*` | `__mul__` | (b.data, a.data) |
| `v**` | `__pow__` | (n * x^(n-1),) |
| `vlog` | `log()` | (1/x,) |
| `vexp` | `exp()` | (exp(x),) |
| `vrelu` | `relu()` | (1.0 if x>0 else 0.0,) |

Derived: `vneg` = `v* v -1`, `v-` = `v+ a (vneg b)`, `v/` = `v* a (v** b -1)`.

`vsum` folds `v+` over a list of Values with `(val 0)` as the initial accumulator.

## `backward`

Topological sort via DFS with a SRFI-69 hash set (keyed by `eq?` identity). The DFS recurses into children before consing the current node, so the resulting `topo` list has the loss at the head and leaves at the tail — exactly the reverse topological order needed for gradient propagation.

Sets `loss.grad = 1`, then walks `topo` with `for-each`, accumulating gradients: `child.grad += local_grad * v.grad`.

## Decisions

- **`ensure-value` for mixed arithmetic**. Python's operator overloading (`__radd__`, `__rmul__`) handles `number + Value` transparently. In Scheme, `v+` and `v*` call `ensure-value` on both arguments to achieve the same. `v**`, `vlog`, `vexp`, and `vrelu` always receive Values, so they don't need it.
- **`eq?` for the visited hash set**. The backward pass needs identity-based comparison (is this the same node?), not structural comparison (do these nodes have the same data?). Multiple Value nodes can have identical data/grad/children but must be tracked separately. Using `(make-hash-table eq?)` matches Python's `set()` which uses object identity.
- **No `hash-by-identity` provided**. SRFI-69 implementations differ on hash function signatures (Chibi expects two arguments, CHICKEN one). The default hash works correctly with `eq?` since `eq?`-equal objects always produce the same hash under any hash function.
- **`cons`-based topological order is naturally reversed**. The DFS appends via `(set! topo (cons v topo))`, which prepends each node. Since children are visited before parents, parents end up at the head. This gives the correct order for gradient propagation without an explicit `reverse` call.
