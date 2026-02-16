# Section 1: Imports

## R7RS-small libraries

- `(scheme base)` — core language
- `(scheme inexact)` — `exp`, `log`, `sqrt`, `acos`, etc.
- `(scheme file)` — `open-input-file`, `file-exists?`
- `(scheme write)` — `display`
- `(scheme char)` — `char-whitespace?`

## SRFIs

- `(srfi 1)` — list library (`fold`, `iota`, `zip`, `filter`, etc.)
- `(srfi 27)` — random numbers (`random-real`, `random-integer`)
- `(srfi 69)` — hash tables (`make-hash-table`, `hash-table-ref`, `hash-table-set!`)
- `(srfi 132)` — sorting (`list-sort`)

## Decisions

- **Dropped SRFI-13** (string libraries). It's not available in Chibi Scheme's default installation. We only needed `string-trim` for stripping whitespace from input lines, which is trivial to implement with `char-whitespace?` from `(scheme char)`.
- **SRFI-69 instead of SRFI-125**. SRFI-125 (Intermediate Hash Tables) is not available as a CHICKEN egg. SRFI-69 (Basic Hash Tables) is supported by both Chibi and CHICKEN and covers all our needs. The only API difference is `hash-table-exists?` (SRFI-69) vs `hash-table-contains?` (SRFI-125).
