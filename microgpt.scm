;;; microgpt.scm — A complete GPT language model in R7RS-small Scheme
;;;
;;; Translated from Karpathy's microgpt.py. This is the complete algorithm
;;; for training and running a Transformer language model. Everything else
;;; — GPUs, parallelism, larger models — is just efficiency.
;;;
;;; The program has five acts:
;;;   1. Data & Tokenization: load names, convert characters to numbers
;;;   2. Autograd Engine: automatic differentiation via computation graphs
;;;   3. The Model: embeddings, attention, MLP — the Transformer architecture
;;;   4. Training: show examples, measure error, compute gradients, update
;;;   5. Inference: generate new names by sampling from the trained model

(import (scheme base)
        (scheme inexact)
        (scheme file)
        (scheme write)
        (scheme char)
        (srfi 1)
        (srfi 27)
        (srfi 69)
        (srfi 132))

;;; ========================================================================
;;; Act 1: Data & Tokenization
;;; ========================================================================
;;; A language model learns patterns from data. The dataset is the source of
;;; truth — the model can only learn what's present in the data.
;;;
;;; We load ~32,000 human names, shuffle them (so the model doesn't learn
;;; ordering artifacts like "all 'a' names come first"), and build a tokenizer
;;; to convert characters to numbers and back.
;;; ========================================================================

(random-source-pseudo-randomize! default-random-source 0 42)

;;; --- PRNG helpers ---
;;; SRFI-27 provides uniform random numbers. On top of these we build the
;;; three random operations microgpt needs.

;; Weight initialization needs Gaussian (bell-curve) random numbers.
;; Small random values (std=0.08) break symmetry (so parameters don't all
;; update identically) while avoiding numerical instability.
;; Box-Muller: convert two uniform samples to one Gaussian sample.
(define (gauss mean std)
  (let ((u1 (random-real))
        (u2 (random-real)))
    (+ mean (* std (sqrt (* -2 (log u1))) (cos (* 2 (acos -1) u2))))))

;; Shuffling the dataset prevents the model from learning spurious patterns
;; based on the order names appear in the file.
;; Fisher-Yates: walk from the end, swap each element with a random earlier one.
(define (vector-shuffle! vec)
  (let ((n (vector-length vec)))
    (do ((i (- n 1) (- i 1)))
        ((<= i 0) vec)
      (let* ((j (random-integer (+ i 1)))
             (tmp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j tmp)))))

(define (shuffle lst)
  (let ((vec (list->vector lst)))
    (vector-shuffle! vec)
    (vector->list vec)))

;; During inference, the model outputs a probability distribution over tokens.
;; We sample from it: higher-probability tokens are more likely to be chosen,
;; but any token has a chance — this is what makes each generation unique.
;; Walks weights accumulating a running sum; returns the index where the
;; cumulative sum exceeds a uniform random threshold.
(define (weighted-choice weights)
  (let* ((total (fold + 0 weights))
         (threshold (* (random-real) total)))
    (let loop ((ws weights) (acc 0) (i 0))
      (if (null? (cdr ws))
          i
          (let ((acc (+ acc (car ws))))
            (if (>= acc threshold)
                i
                (loop (cdr ws) acc (+ i 1))))))))

;;; --- Dataset loading ---

(define (string-trim s)
  (let* ((chars (string->list s))
         (trimmed (drop-while char-whitespace? chars))
         (trimmed (reverse (drop-while char-whitespace? (reverse trimmed)))))
    (list->string trimmed)))

(define (read-lines filename)
  (call-with-input-file filename
    (lambda (port)
      (let loop ((lines '()))
        (let ((line (read-line port)))
          (if (eof-object? line)
              (reverse lines)
              (let ((trimmed (string-trim line)))
                (if (string=? trimmed "")
                    (loop lines)
                    (loop (cons trimmed lines))))))))))

(define docs (list->vector (shuffle (read-lines "input.txt"))))
(display "num docs: ") (display (vector-length docs)) (newline)

;;; --- Tokenizer ---
;;; Computers can't do math on characters — we need numbers. The tokenizer
;;; maps each unique character to an integer ID: 'a'→0, 'b'→1, ..., 'z'→25.
;;; This is a character-level tokenizer (each character is one token).
;;; Production models like GPT-2 use BPE with 50,000+ tokens instead.

;; Collect all unique characters, sort them, store as a vector for O(1) decoding
(define uchars
  (list->vector
    (list-sort char<?
      (delete-duplicates
        (append-map string->list (vector->list docs))
        char=?))))

;; Hash table for O(1) encoding (char → token id)
(define char->id
  (let ((ht (make-hash-table)))
    (do ((i 0 (+ i 1))) ((= i (vector-length uchars)) ht)
      (hash-table-set! ht (vector-ref uchars i) i))))

;; BOS (Beginning of Sequence) is a special token that marks both the start
;; and end of a name. For "emma", the training sequence is:
;;   [BOS, 'e', 'm', 'm', 'a', BOS]
;; This teaches the model when to start generating and when to stop.
(define BOS (vector-length uchars))
(define vocab-size (+ (vector-length uchars) 1))
(display "vocab size: ") (display vocab-size) (newline)

;;; ========================================================================
;;; Act 2: The Autograd Engine
;;; ========================================================================
;;; Training requires knowing how to adjust each parameter to reduce error.
;;; The naive approach — try random changes — is hopelessly slow with 4000+
;;; parameters. Derivatives tell us the direction and magnitude of the best
;;; adjustment for every parameter simultaneously.
;;;
;;; The trick: every math operation records what it did (children) and how to
;;; differentiate it (local gradients). The forward pass builds a computation
;;; graph as a side effect. The backward pass walks this graph in reverse,
;;; applying the chain rule to compute all derivatives at once.
;;;
;;; Without autograd: 4065 forward passes to estimate 4064 gradients.
;;; With autograd: 1 forward + 1 backward = 2 passes. ~2000x more efficient.
;;; ========================================================================

;; Each Value node in the computation graph stores four things:
;;   data:        the scalar value computed during the forward pass
;;   grad:        d(loss)/d(this node), filled during backward (starts at 0)
;;   children:    the input nodes that produced this node
;;   local-grads: d(this node)/d(each child) — the chain rule factors
(define-record-type <value>
  (make-value data grad children local-grads)
  value?
  (data value-data set-value-data!)
  (grad value-grad set-value-grad!)
  (children value-children)
  (local-grads value-local-grads))

;; Leaf node: a plain number with no children (e.g., a model parameter)
(define (val x) (make-value x 0 '() '()))

;; Auto-wrap plain numbers so arithmetic operations accept mixed arguments
(define (ensure-value x) (if (value? x) x (val x)))

;; Each operation creates a new Value recording its inputs and local derivatives.
;; The local derivative answers: "if I nudge this child, how much does the
;; output change?" The chain rule composes these into the full derivative.

;; Addition: d(a+b)/da = 1, d(a+b)/db = 1
(define (v+ a b)
  (let ((a (ensure-value a)) (b (ensure-value b)))
    (make-value (+ (value-data a) (value-data b)) 0
                (list a b) (list 1 1))))

;; Multiplication: d(a*b)/da = b, d(a*b)/db = a (the "swap" rule)
(define (v* a b)
  (let ((a (ensure-value a)) (b (ensure-value b)))
    (make-value (* (value-data a) (value-data b)) 0
                (list a b) (list (value-data b) (value-data a)))))

;; Power: d(x^n)/dx = n * x^(n-1). Exponent n is a plain number.
(define (v** v n)
  (make-value (expt (value-data v) n) 0
              (list v) (list (* n (expt (value-data v) (- n 1))))))

;; Log: d(ln x)/dx = 1/x. Used in the cross-entropy loss: -log(P(correct)).
(define (vlog v)
  (make-value (log (value-data v)) 0
              (list v) (list (/ 1 (value-data v)))))

;; Exp: d(e^x)/dx = e^x. Used in softmax to make all values positive.
(define (vexp v)
  (make-value (exp (value-data v)) 0
              (list v) (list (exp (value-data v)))))

;; ReLU: the non-linear activation function. Without non-linearity, stacking
;; linear layers collapses to a single linear layer — the model couldn't learn
;; complex patterns. ReLU passes positive values through, zeros out negatives.
;; d(relu x)/dx = 1 if x > 0, else 0.
(define (vrelu v)
  (make-value (max 0 (value-data v)) 0
              (list v) (list (if (> (value-data v) 0) 1.0 0.0))))

;; Derived operations built from primitives
(define (vneg v) (v* v -1))
(define (v- a b) (v+ a (vneg b)))
(define (v/ a b) (v* a (v** (ensure-value b) -1)))

(define (vsum lst) (fold v+ (val 0) lst))

;; Backward pass: compute d(loss)/d(node) for every node in the graph.
;; 1. Topological sort via DFS — ensures children are processed before parents
;; 2. Seed: loss.grad = 1 (the derivative of loss with respect to itself)
;; 3. Propagate: for each node, child.grad += local_grad * node.grad
;; The "+=" is crucial: a value used multiple times accumulates gradients
;; from all paths through the graph.
(define (backward loss)
  (let ((topo '())
        (visited (make-hash-table eq?)))
    (let build-topo ((v loss))
      (when (not (hash-table-exists? visited v))
        (hash-table-set! visited v #t)
        (for-each build-topo (value-children v))
        (set! topo (cons v topo))))
    (set-value-grad! loss 1)
    (for-each
      (lambda (v)
        (for-each
          (lambda (child local-grad)
            (set-value-grad! child
              (+ (value-grad child) (* local-grad (value-grad v)))))
          (value-children v)
          (value-local-grads v)))
      topo)))

;;; ========================================================================
;;; Act 3: The Model — a Transformer
;;; ========================================================================
;;; The architecture maps a token and its position to a probability
;;; distribution over what comes next. It follows GPT-2's structure:
;;;
;;;   Embed → Normalize → Attend → Think → Predict
;;;
;;; - Embeddings give each token a rich 16-dimensional representation
;;; - Attention lets tokens look at context ("what came before me?")
;;; - MLP does non-linear processing ("what patterns do I see?")
;;; - Residual connections preserve information (add, don't replace)
;;; - RMSNorm keeps values well-scaled (prevents exploding/vanishing)
;;;
;;; This function IS the Transformer. Same architecture as GPT-2/3/4,
;;; just with smaller matrices.
;;; ========================================================================

;;; --- Hyperparameters ---
;;; These are choices the programmer makes. The model learns everything else.
;;; Compare: GPT-2 uses n_embd=768, n_head=12, n_layer=12, block_size=1024.

(define n-layer 1)       ; depth: number of attention+MLP blocks
(define n-embd 16)       ; width: dimensions per token representation
(define block-size 16)   ; max context length (longest name is 15 chars)
(define n-head 4)        ; number of independent attention perspectives
(define head-dim (/ n-embd n-head)) ; dimensions per head (16/4 = 4)

;;; --- Weight initialization ---
;;; Before training, all parameters are random. After training, they encode
;;; everything the model has "learned." Small Gaussian values (std=0.08)
;;; break symmetry while avoiding numerical instability.

(define (matrix nout nin)
  (let ((rows (make-vector nout '())))
    (do ((i 0 (+ i 1))) ((= i nout) rows)
      (vector-set! rows i
        (map (lambda (_) (val (gauss 0 0.08))) (iota nin))))))

;;; --- State dictionary ---
;;; All the model's knowledge lives in these matrices:
;;;   wte (27x16):  token embeddings — what each character "means"
;;;   wpe (16x16):  position embeddings — what each position "means"
;;;   lm_head (27x16): output projection — convert hidden state to predictions
;;;   Per layer:
;;;     attn_wq/wk/wv (16x16): query/key/value projections for attention
;;;     attn_wo (16x16): output projection mixing attention heads
;;;     mlp_fc1 (64x16): MLP expansion (16→64, 4x wider for richer processing)
;;;     mlp_fc2 (16x64): MLP compression (64→16, back to embedding size)

(define state-dict
  (let ((sd (make-hash-table string=? string-hash)))
    (hash-table-set! sd "wte" (matrix vocab-size n-embd))
    (hash-table-set! sd "wpe" (matrix block-size n-embd))
    (hash-table-set! sd "lm_head" (matrix vocab-size n-embd))
    (do ((i 0 (+ i 1))) ((= i n-layer) sd)
      (let ((prefix (string-append "layer" (number->string i) ".")))
        (hash-table-set! sd (string-append prefix "attn_wq") (matrix n-embd n-embd))
        (hash-table-set! sd (string-append prefix "attn_wk") (matrix n-embd n-embd))
        (hash-table-set! sd (string-append prefix "attn_wv") (matrix n-embd n-embd))
        (hash-table-set! sd (string-append prefix "attn_wo") (matrix n-embd n-embd))
        (hash-table-set! sd (string-append prefix "mlp_fc1") (matrix (* 4 n-embd) n-embd))
        (hash-table-set! sd (string-append prefix "mlp_fc2") (matrix n-embd (* 4 n-embd)))))))

;; Flatten all weight matrices into one list for the optimizer.
;; The optimizer treats every parameter identically — it just needs to
;; iterate over all of them to apply gradient updates.
(define params
  (append-map
    (lambda (key)
      (let ((mat (hash-table-ref state-dict key)))
        (apply append (vector->list mat))))
    (hash-table-keys state-dict)))

(display "num params: ") (display (length params)) (newline)

;;; --- Architecture functions ---

;; Linear layer: the fundamental operation for mixing information.
;; Each output is a weighted sum (dot product) of all inputs. The weights
;; determine "how much of each input goes into each output."
;; y = Wx, where W is the weight matrix and x is the input vector.
(define (linear x w)
  (map (lambda (wo) (vsum (map v* wo x))) (vector->list w)))

;; Softmax: convert raw logits (any real numbers) into a probability
;; distribution (all positive, sum to 1). The exponential amplifies
;; differences — the largest logit gets the largest probability.
;; Subtracting max before exp prevents overflow (e^1000 = infinity!).
;; The subtraction doesn't change the result: softmax(z-c) = softmax(z).
(define (softmax logits)
  (let* ((max-val (fold max -inf.0 (map value-data logits)))
         (exps (map (lambda (v) (vexp (v+ v (- max-val)))) logits))
         (total (vsum exps)))
    (map (lambda (e) (v/ e total)) exps)))

;; RMSNorm: keeps values well-scaled as data flows through layers.
;; Without normalization, numbers can drift (become huge or tiny),
;; causing overflow in exp or vanishing gradients. RMSNorm divides
;; each element by the root-mean-square magnitude of the vector.
;; The epsilon (1e-5) prevents division by zero.
(define (rmsnorm x)
  (let* ((ms (v/ (vsum (map (lambda (xi) (v* xi xi)) x)) (length x)))
         (scale (v** (v+ ms 1e-5) -0.5)))
    (map (lambda (xi) (v* xi scale)) x)))

;; List slicing helper for splitting Q/K/V into per-head chunks
(define (list-slice lst start len)
  (take (list-tail lst start) len))

;; The GPT forward pass: the complete Transformer.
;; Takes a token ID and position, returns 27 logits (one score per character).
;; These logits are NOT probabilities yet — softmax converts them later.
;;
;; Data flow:
;;   token_id → wte lookup (what) ─┐
;;   pos_id   → wpe lookup (where) ┘→ add → rmsnorm
;;     → [attention block + residual] → [MLP block + residual] → lm_head → logits
(define (gpt token-id pos-id keys vals)
  ;; Step 1: Embed — look up token and position embeddings (O(1) table lookup),
  ;; then add them element-wise. Addition works because both live in the same
  ;; 16-dimensional space. The token embedding captures "what" (the character's
  ;; meaning), the position embedding captures "where" (its position matters:
  ;; 'e' at position 0 behaves very differently from 'e' at position 4).
  (let* ((tok-emb (vector-ref (hash-table-ref state-dict "wte") token-id))
         (pos-emb (vector-ref (hash-table-ref state-dict "wpe") pos-id))
         (x (map v+ tok-emb pos-emb))
         (x (rmsnorm x)))
    (let layer-loop ((li 0) (x x))
      (if (= li n-layer)
          ;; Final output: project from 16 dims to 27 (one score per token).
          ;; The highest-scoring token is the model's best guess for what
          ;; comes next.
          (linear x (hash-table-ref state-dict "lm_head"))
          (let* ((prefix (string-append "layer" (number->string li) "."))

                 ;; === Multi-head Attention block ===
                 ;; The core question: "given what I've seen so far, what
                 ;; information from past tokens is relevant to predicting
                 ;; the next one?"
                 ;;
                 ;; Each token produces three vectors:
                 ;;   Q (query):  "what am I looking for?"
                 ;;   K (key):    "what do I contain?"
                 ;;   V (value):  "here's my actual information"
                 ;;
                 ;; Attention score = Q dot K / sqrt(d) measures relevance.
                 ;; Softmax converts scores to weights (a probability over
                 ;; past tokens). Output = weighted sum of V vectors.

                 ;; Save input for residual connection (add, don't replace)
                 (x-residual x)
                 (x (rmsnorm x))
                 (q (linear x (hash-table-ref state-dict (string-append prefix "attn_wq"))))
                 (k (linear x (hash-table-ref state-dict (string-append prefix "attn_wk"))))
                 (v (linear x (hash-table-ref state-dict (string-append prefix "attn_wv")))))

            ;; KV cache: store this position's key and value for future tokens.
            ;; When processing token 5, keys/values from tokens 0-4 are already
            ;; cached — no recomputation needed.
            (vector-set! keys li (append (vector-ref keys li) (list k)))
            (vector-set! vals li (append (vector-ref vals li) (list v)))

            (let* ((cached-keys (vector-ref keys li))
                   (cached-vals (vector-ref vals li))

                   ;; Multi-head attention: split Q/K/V into n-head independent
                   ;; groups, each operating on head-dim dimensions. Different
                   ;; heads learn different "perspectives" — one might focus on
                   ;; adjacent characters, another on the first character, etc.
                   ;; Results are concatenated back to n-embd dimensions.
                   (x-attn
                     (let head-loop ((h 0) (acc '()))
                       (if (= h n-head)
                           acc
                           (let* ((hs (* h head-dim))
                                  (q-h (list-slice q hs head-dim))
                                  (k-h (map (lambda (ki) (list-slice ki hs head-dim)) cached-keys))
                                  (v-h (map (lambda (vi) (list-slice vi hs head-dim)) cached-vals))
                                  ;; Scaled dot-product attention.
                                  ;; Dividing by sqrt(head-dim) prevents scores from
                                  ;; becoming too large, which would make softmax
                                  ;; saturate (one weight ≈ 1, all others ≈ 0).
                                  (scale (/ 1.0 (sqrt head-dim)))
                                  (attn-logits
                                    (map (lambda (k-t)
                                           (v* (vsum (map v* q-h k-t)) scale))
                                         k-h))
                                  (attn-weights (softmax attn-logits))
                                  ;; Blend value vectors weighted by relevance
                                  (head-out
                                    (fold (lambda (wt vt acc)
                                            (map (lambda (vi ai) (v+ (v* wt vi) ai)) vt acc))
                                          (make-list head-dim (val 0))
                                          attn-weights v-h)))
                             (head-loop (+ h 1) (append acc head-out))))))

                   ;; Output projection: mix information from all heads
                   (x (linear x-attn (hash-table-ref state-dict (string-append prefix "attn_wo"))))
                   ;; Residual connection: output = attention_output + original_input.
                   ;; If attention learns nothing useful, it outputs zeros, and
                   ;; the input passes through unchanged. This also creates a
                   ;; "gradient highway" — d(a+b)/da = 1, so gradients flow
                   ;; straight through without shrinking.
                   (x (map v+ x x-residual))

                   ;; === MLP block ===
                   ;; Attention is linear (weighted sums). To learn complex
                   ;; patterns like "after 'qu', a vowel usually follows," the
                   ;; model needs non-linear processing.
                   ;;
                   ;; The MLP is: expand (16→64) → ReLU → compress (64→16).
                   ;; The 4x expansion gives more room for computation.
                   ;; ReLU is the non-linearity that makes this more powerful
                   ;; than a single linear layer.
                   (x-residual x)
                   (x (rmsnorm x))
                   (x (linear x (hash-table-ref state-dict (string-append prefix "mlp_fc1"))))
                   (x (map vrelu x))
                   (x (linear x (hash-table-ref state-dict (string-append prefix "mlp_fc2"))))
                   ;; Residual connection for MLP block
                   (x (map v+ x x-residual)))
              (layer-loop (+ li 1) x)))))))

;;; ========================================================================
;;; Act 4: Training
;;; ========================================================================
;;; The model starts knowing nothing — all 4192 parameters are random.
;;; Training is a loop: show the model a name, measure how wrong it was
;;; (loss), compute which direction to nudge each parameter (backward),
;;; and nudge them (Adam optimizer).
;;;
;;; Loss starts near ln(27) ≈ 3.30 (random guessing among 27 tokens)
;;; and decreases as the model learns patterns in names.
;;; ========================================================================

;;; --- Adam optimizer setup ---
;;; Adam (Adaptive Moment Estimation) improves on basic gradient descent
;;; in three ways:
;;;   1. Momentum (m): smooths noisy gradients by keeping a running average
;;;   2. Adaptive rate (v): parameters with volatile gradients get smaller
;;;      steps (cautious), stable ones get larger steps (confident)
;;;   3. Bias correction: compensates for m and v starting at zero

(define learning-rate 0.01)
(define beta1 0.85)   ; momentum decay: keep 85% of previous momentum
(define beta2 0.99)   ; variance decay: keep 99% of previous variance
(define eps-adam 1e-8) ; prevent division by zero

(define num-params (length params))
(define params-vec (list->vector params))
(define adam-m (make-vector num-params 0.0)) ; first moment (momentum)
(define adam-v (make-vector num-params 0.0)) ; second moment (variance)

;;; --- Tokenization helper ---
;;; Wraps a name with BOS on both sides. For "emma":
;;;   [BOS, 'e', 'm', 'm', 'a', BOS]
;;; The model learns from pairs: BOS→'e', 'e'→'m', ..., 'a'→BOS.
;;; The final BOS teaches the model when names should end.

(define (tokenize doc)
  (list->vector
    (append (list BOS)
            (map (lambda (ch) (hash-table-ref char->id ch))
                 (string->list doc))
            (list BOS))))

;;; --- Training loop ---

(define num-steps 1000)

(do ((step 0 (+ step 1))) ((= step num-steps))
  (let* ((doc (vector-ref docs (modulo step (vector-length docs))))
         (tokens (tokenize doc))
         (n (min block-size (- (vector-length tokens) 1)))
         ;; Fresh KV cache for each document — each name is independent
         (keys (make-vector n-layer '()))
         (vals (make-vector n-layer '())))

    ;; Forward pass: for each position, predict the next token.
    ;; This builds the computation graph — every Value node remembers
    ;; how it was produced, enabling the backward pass.
    (let pos-loop ((pos-id 0) (losses '()))
      (if (= pos-id n)
          ;; All positions processed — compute mean loss and update
          (let* ((loss (v* (vsum (reverse losses)) (/ 1.0 n))))

            ;; Backward pass: one call computes d(loss)/d(parameter) for
            ;; all 4192 parameters simultaneously. Each parameter's .grad
            ;; field now tells us: "nudge me in this direction to reduce loss."
            (backward loss)

            ;; Adam update: for each parameter, use its gradient to update
            ;; its value. Linear learning rate decay: large steps early for
            ;; fast learning, small steps late for fine-tuning.
            (let ((lr-t (* learning-rate (- 1 (/ step num-steps)))))
              (do ((i 0 (+ i 1))) ((= i num-params))
                (let* ((p (vector-ref params-vec i))
                       (g (value-grad p))
                       ;; Momentum: smooth the gradient over time
                       (mi (+ (* beta1 (vector-ref adam-m i))
                              (* (- 1 beta1) g)))
                       ;; Variance: track how noisy this parameter's gradient is
                       (vi (+ (* beta2 (vector-ref adam-v i))
                              (* (- 1 beta2) (* g g))))
                       ;; Bias correction: inflate early estimates (m,v start at 0)
                       (m-hat (/ mi (- 1 (expt beta1 (+ step 1)))))
                       (v-hat (/ vi (- 1 (expt beta2 (+ step 1))))))
                  (vector-set! adam-m i mi)
                  (vector-set! adam-v i vi)
                  ;; The update: p -= lr * momentum / sqrt(variance)
                  ;; High variance → smaller step (cautious)
                  ;; Low variance → larger step (confident)
                  (set-value-data! p
                    (- (value-data p) (* lr-t (/ m-hat (+ (sqrt v-hat) eps-adam)))))
                  ;; Reset gradient — backward() uses +=, so without reset
                  ;; gradients from previous steps would corrupt this step.
                  (set-value-grad! p 0))))

            (display "step ") (display (+ step 1))
            (display " / ") (display num-steps)
            (display " | loss ") (display (value-data loss))
            (newline))

          ;; Process one position: predict next token, measure error.
          ;; Cross-entropy loss = -log(P(correct token)).
          ;; If the model assigns 90% to the right answer: loss = 0.105 (good).
          ;; If only 1%: loss = 4.605 (bad — heavily punishes confident errors).
          (let* ((token-id (vector-ref tokens pos-id))
                 (target-id (vector-ref tokens (+ pos-id 1)))
                 (logits (gpt token-id pos-id keys vals))
                 (probs (softmax logits))
                 (loss-t (vneg (vlog (list-ref probs target-id)))))
            (pos-loop (+ pos-id 1) (cons loss-t losses)))))))

;;; ========================================================================
;;; Act 5: Inference — the model speaks
;;; ========================================================================
;;; The 4192 parameters have been tuned over 1000 steps. Now we use the
;;; trained model to generate names it has never seen.
;;;
;;; Generation is autoregressive: each output token becomes the input for
;;; the next step. Start with BOS, predict one character at a time, stop
;;; when the model predicts BOS again ("this name is done").
;;; ========================================================================

;; Temperature controls "creativity." It divides logits before softmax:
;;   temperature < 1: amplifies differences → sharper distribution → more
;;     predictable (strongly favors the top choice)
;;   temperature = 1: standard probabilities
;;   temperature > 1: dampens differences → flatter distribution → more
;;     random (weaker choices get better odds)
;; Named after statistical mechanics: low temperature = order, high = chaos.
(define temperature 0.5)

(newline)
(display "--- inference (new, hallucinated names) ---")
(newline)

(do ((sample-idx 0 (+ sample-idx 1))) ((= sample-idx 20))
  (let ((keys (make-vector n-layer '()))
        (vals (make-vector n-layer '())))
    (let sample-loop ((pos-id 0) (token-id BOS) (chars '()))
      (if (= pos-id block-size)
          (begin
            (display "sample ") (display (+ sample-idx 1))
            (display ": ") (display (list->string (reverse chars)))
            (newline))
          (let* ((logits (gpt token-id pos-id keys vals))
                 (scaled (map (lambda (l) (v/ l temperature)) logits))
                 (probs (softmax scaled))
                 (weights (map value-data probs))
                 (next-id (weighted-choice weights)))
            (if (= next-id BOS)
                ;; BOS predicted — model says "this name is done"
                (begin
                  (display "sample ") (display (+ sample-idx 1))
                  (display ": ") (display (list->string (reverse chars)))
                  (newline))
                ;; Append character, feed it back as input (autoregressive)
                (sample-loop (+ pos-id 1) next-id
                             (cons (vector-ref uchars next-id) chars))))))))
