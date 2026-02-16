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
  (let ((total (fold + 0 weights))
        (threshold (* (random-real) (fold + 0 weights))))
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

(define docs (shuffle (read-lines "input.txt")))
(display "num docs: ") (display (length docs)) (newline)

;;; --- Section 4: Tokenizer ---

;; Unique sorted characters from all docs (vector for O(1) decoding)
(define uchars
  (list->vector
    (list-sort char<?
      (delete-duplicates
        (append-map string->list docs)
        char=?))))

;; Reverse lookup: char → token id (O(1) via hash table)
(define char->id
  (let ((ht (make-hash-table)))
    (do ((i 0 (+ i 1))) ((= i (vector-length uchars)) ht)
      (hash-table-set! ht (vector-ref uchars i) i))))

(define BOS (vector-length uchars))
(define vocab-size (+ (vector-length uchars) 1))
(display "vocab size: ") (display vocab-size) (newline)
