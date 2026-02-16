(import (scheme base)
        (scheme inexact)
        (scheme file)
        (scheme write)
        (scheme char)
        (srfi 1)
        (srfi 27)
        (srfi 69)
        (srfi 132))

;;; --- Section 2: PRNG helpers ---

;; Box-Muller transform: generate Gaussian random numbers from uniform samples
(define (gauss mean std)
  (let ((u1 (random-real))
        (u2 (random-real)))
    (+ mean (* std (sqrt (* -2 (log u1))) (cos (* 2 (acos -1) u2))))))

;; Fisher-Yates shuffle: in-place shuffle of a vector
(define (vector-shuffle! vec)
  (let ((n (vector-length vec)))
    (do ((i (- n 1) (- i 1)))
        ((<= i 0) vec)
      (let* ((j (random-integer (+ i 1)))
             (tmp (vector-ref vec i)))
        (vector-set! vec i (vector-ref vec j))
        (vector-set! vec j tmp)))))

;; Shuffle a list by converting to vector, shuffling, and converting back
(define (shuffle lst)
  (let ((vec (list->vector lst)))
    (vector-shuffle! vec)
    (vector->list vec)))

;; Weighted random choice: pick an index from a list of weights
;; Walks the weights accumulating a running sum, returns the index
;; where the cumulative sum exceeds a uniform random threshold.
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

;;; --- Section 3: Dataset loading ---

;; string-trim: strip leading and trailing whitespace (replaces SRFI-13)
(define (string-trim s)
  (let* ((chars (string->list s))
         (trimmed (drop-while char-whitespace? chars))
         (trimmed (reverse (drop-while char-whitespace? (reverse trimmed)))))
    (list->string trimmed)))

;; Read all non-empty lines from a file, trimming whitespace
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

;;; --- Section 4: Tokenizer ---

;; Unique sorted characters from all docs (vector for O(1) decoding)
(define uchars
  (list->vector
    (list-sort char<?
      (delete-duplicates
        (append-map string->list (vector->list docs))
        char=?))))

;; Reverse lookup: char → token id (O(1) via hash table)
(define char->id
  (let ((ht (make-hash-table)))
    (do ((i 0 (+ i 1))) ((= i (vector-length uchars)) ht)
      (hash-table-set! ht (vector-ref uchars i) i))))

(define BOS (vector-length uchars))
(define vocab-size (+ (vector-length uchars) 1))
(display "vocab size: ") (display vocab-size) (newline)

;;; --- Section 5: Value record type + autograd operations + backward ---

;; Autograd node: a scalar value in a computation graph
(define-record-type <value>
  (make-value data grad children local-grads)
  value?
  (data value-data set-value-data!)
  (grad value-grad set-value-grad!)
  (children value-children)
  (local-grads value-local-grads))

;; Wrap a plain number as a leaf Value node
(define (val x) (make-value x 0 '() '()))

;; Auto-wrap plain numbers so v+ and v* can accept mixed arguments
(define (ensure-value x) (if (value? x) x (val x)))

;; Addition: d(a+b)/da = 1, d(a+b)/db = 1
(define (v+ a b)
  (let ((a (ensure-value a)) (b (ensure-value b)))
    (make-value (+ (value-data a) (value-data b)) 0
                (list a b) (list 1 1))))

;; Multiplication: d(a*b)/da = b, d(a*b)/db = a
(define (v* a b)
  (let ((a (ensure-value a)) (b (ensure-value b)))
    (make-value (* (value-data a) (value-data b)) 0
                (list a b) (list (value-data b) (value-data a)))))

;; Power: d(x^n)/dx = n * x^(n-1). Exponent n is a plain number.
(define (v** v n)
  (make-value (expt (value-data v) n) 0
              (list v) (list (* n (expt (value-data v) (- n 1))))))

;; Log: d(log x)/dx = 1/x
(define (vlog v)
  (make-value (log (value-data v)) 0
              (list v) (list (/ 1 (value-data v)))))

;; Exp: d(exp x)/dx = exp(x)
(define (vexp v)
  (make-value (exp (value-data v)) 0
              (list v) (list (exp (value-data v)))))

;; ReLU: d(relu x)/dx = 1 if x > 0, else 0
(define (vrelu v)
  (make-value (max 0 (value-data v)) 0
              (list v) (list (if (> (value-data v) 0) 1.0 0.0))))

;; Derived operations
(define (vneg v) (v* v -1))
(define (v- a b) (v+ a (vneg b)))
(define (v/ a b) (v* a (v** b -1)))

;; Sum a list of Values
(define (vsum lst) (fold v+ (val 0) lst))

;; Backward pass: topological sort + gradient propagation
(define (backward loss)
  ;; Build topological order via DFS (cons prepends, so topo ends up
  ;; with loss at the head and leaves at the tail — the order we need)
  (let ((topo '())
        (visited (make-hash-table eq?)))
    (let build-topo ((v loss))
      (when (not (hash-table-exists? visited v))
        (hash-table-set! visited v #t)
        (for-each build-topo (value-children v))
        (set! topo (cons v topo))))
    ;; Propagate gradients from loss back to all parameters
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

;;; --- Section 6: Model parameters ---

(define n-layer 1)
(define n-embd 16)
(define block-size 16)
(define n-head 4)
(define head-dim (/ n-embd n-head))

;; Weight matrix: vector of nout rows, each row a list of nin random Values
(define (matrix nout nin)
  (let ((rows (make-vector nout '())))
    (do ((i 0 (+ i 1))) ((= i nout) rows)
      (vector-set! rows i
        (map (lambda (_) (val (gauss 0 0.08))) (iota nin))))))

;; State dict: string-keyed hash table of weight matrices
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

;; Flatten all parameters into a single list of Value nodes
(define params
  (append-map
    (lambda (key)
      (let ((mat (hash-table-ref state-dict key)))
        (apply append (vector->list mat))))
    (hash-table-keys state-dict)))

(display "num params: ") (display (length params)) (newline)

;;; --- Section 7: Model architecture ---

;; Linear layer: matrix-vector multiply (each row dot x)
(define (linear x w)
  (map (lambda (wo) (vsum (map v* wo x))) (vector->list w)))

;; Softmax: subtract max for numerical stability, exp, normalize
(define (softmax logits)
  (let* ((max-val (fold max -inf.0 (map value-data logits)))
         (exps (map (lambda (v) (vexp (v+ v (- max-val)))) logits))
         (total (vsum exps)))
    (map (lambda (e) (v/ e total)) exps)))

;; RMSNorm: root mean square normalization
(define (rmsnorm x)
  (let* ((ms (v/ (vsum (map (lambda (xi) (v* xi xi)) x)) (length x)))
         (scale (v** (v+ ms 1e-5) -0.5)))
    (map (lambda (xi) (v* xi scale)) x)))

;; Slice a list: return elements from index start to start+len
(define (list-slice lst start len)
  (take (list-tail lst start) len))

;; GPT forward pass: token + position embedding, attention, MLP, output logits
(define (gpt token-id pos-id keys vals)
  (let* ((tok-emb (vector-ref (hash-table-ref state-dict "wte") token-id))
         (pos-emb (vector-ref (hash-table-ref state-dict "wpe") pos-id))
         (x (map v+ tok-emb pos-emb))
         (x (rmsnorm x)))
    ;; Layer loop (named let threads x through iterations)
    (let layer-loop ((li 0) (x x))
      (if (= li n-layer)
          ;; After all layers, compute output logits
          (linear x (hash-table-ref state-dict "lm_head"))
          (let* ((prefix (string-append "layer" (number->string li) "."))
                 ;; 1) Multi-head Attention block
                 (x-residual x)
                 (x (rmsnorm x))
                 (q (linear x (hash-table-ref state-dict (string-append prefix "attn_wq"))))
                 (k (linear x (hash-table-ref state-dict (string-append prefix "attn_wk"))))
                 (v (linear x (hash-table-ref state-dict (string-append prefix "attn_wv")))))
            ;; Append k and v to cache
            (vector-set! keys li (append (vector-ref keys li) (list k)))
            (vector-set! vals li (append (vector-ref vals li) (list v)))
            (let* ((cached-keys (vector-ref keys li))
                   (cached-vals (vector-ref vals li))
                   ;; Multi-head attention: concatenate head outputs
                   (x-attn
                     (let head-loop ((h 0) (acc '()))
                       (if (= h n-head)
                           acc
                           (let* ((hs (* h head-dim))
                                  (q-h (list-slice q hs head-dim))
                                  (k-h (map (lambda (ki) (list-slice ki hs head-dim)) cached-keys))
                                  (v-h (map (lambda (vi) (list-slice vi hs head-dim)) cached-vals))
                                  ;; Attention logits: dot(q_h, k_t) / sqrt(head_dim)
                                  (scale (/ 1.0 (sqrt head-dim)))
                                  (attn-logits
                                    (map (lambda (k-t)
                                           (v* (vsum (map v* q-h k-t)) scale))
                                         k-h))
                                  (attn-weights (softmax attn-logits))
                                  ;; Weighted sum of value vectors (iterate by position, not dimension)
                                  (head-out
                                    (fold (lambda (wt vt acc)
                                            (map (lambda (vi ai) (v+ (v* wt vi) ai)) vt acc))
                                          (make-list head-dim (val 0))
                                          attn-weights v-h)))
                             (head-loop (+ h 1) (append acc head-out))))))
                   ;; Project attention output
                   (x (linear x-attn (hash-table-ref state-dict (string-append prefix "attn_wo"))))
                   (x (map v+ x x-residual))
                   ;; 2) MLP block
                   (x-residual x)
                   (x (rmsnorm x))
                   (x (linear x (hash-table-ref state-dict (string-append prefix "mlp_fc1"))))
                   (x (map vrelu x))
                   (x (linear x (hash-table-ref state-dict (string-append prefix "mlp_fc2"))))
                   (x (map v+ x x-residual)))
              (layer-loop (+ li 1) x)))))))
