;;;; cl-textclassifier.lisp

(in-package #:cl-textclassifier)

;;; "cl-textclassifier" goes here. Hacks and glory await!

(defun keys (table)
  (let ((acc '()))
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (push k acc))
             table)
    acc))

(defun vals (table)
  (let ((acc '()))
    (maphash #'(lambda (k v)
                 (declare (ignore k))
                 (push v acc))
             table)
    acc))

(defun n-grams (words N)
  "Compute n-grams from WORDS. The end of words is chopped off."
  (delete-if #'null
             (maplist #'(lambda (words)
                          (if (nthcdr (1- N) words)
                              (subseq words 0 N)))
                      words)))

(defun osb (words N)
  "Compute Orthogonal Sparce Bigrams (OSB) from WORDS. The end of
  words is chopped off."
  (delete-if #'null
             (let ((acc '()))
               (mapl #'(lambda (words)
                         (if (nthcdr (1- N) words)
                             (dotimes (i (1- N))
                               (push (cons (car words)
                                           (nth i (cdr words)))
                                     acc))))
                     words)
               acc)))

(defun bag-of-words (words)
  "Generate the bag of words features of WORDS. This is WORDS itself."
  words
  )

(defun default-tokenize-fun (text)
  "Return a list of tokens generated from TEXT. This function uses the
  default langutils tokenizer split-sequence with empty subseq
  removal."
  (multiple-value-bind (success-p idx tokenized-string)
      (langutils:tokenize-string text)
    (declare (ignore success-p idx))
    (split-sequence:split-sequence #\Space
                                   tokenized-string
                                   :remove-empty-subseqs t)))

;; ;; length normalized tf - performs poorly
;; (defun tf-train (vec &key idf)
;;   "Compute term frequencies for VEC, where VEC is, for example, the
;;   tokenized text of a document."
;;   (let ((histogram (make-hash-table :test #'equal))
;;         (denom 0))
;;     (dolist (feature vec)
;;       (incf (gethash feature histogram 0)))
;;     (maphash #'(lambda (k v)
;;                  (setf (gethash k histogram)
;;                        (* (log (1+ v))
;;                           (gethash k idf 1)))
;;                  (incf denom (expt (gethash k histogram) 2)))
;;              histogram)
;;     (setf denom (sqrt denom))
;;     (maphash #'(lambda (k v)
;;                  (setf (gethash k histogram)
;;                        (/ v denom)))
;;              histogram)
;;     histogram))

;; ;; tf with incorrect length normalization (denominator is computed
;; ;; before IDF transformation) - performs surprisingly well
;; (defun tf-train (vec &key idf)
;;   "Compute term frequencies for VEC, where VEC is, for example, the
;;   tokenized text of a document."
;;   (let ((histogram (make-hash-table :test #'equal))
;;         (denom 0))
;;     (dolist (feature vec)
;;       (incf (gethash feature histogram 0)))
;;     (maphash #'(lambda (k v)
;;                  (setf (gethash k histogram)
;;                        (log (1+ v)))
;;                  (incf denom (expt (gethash k histogram) 2)))
;;              histogram)
;;     (setf denom (sqrt denom))
;;     (maphash #'(lambda (k v)
;;                  (setf (gethash k histogram)
;;                        (* (gethash k idf 1) ; TODO-maybe: change 1 to N
;;                           (/ v denom))))
;;              histogram)
;;     histogram))

;; tf without length normalization - slightly improves recall at
;; the expense of precision as compared to tf-good, but results in
;; higher F1 metrics than improper length normalization
(defun tf-train (vec &key idf)
  "Compute term frequencies for VEC, where VEC is, for example, the
  tokenized text of a document."
  (let ((histogram (make-hash-table :test #'equal)))
    (dolist (feature vec)
      (incf (gethash feature histogram 0)))
    (maphash #'(lambda (k v)
                 (setf (gethash k histogram)
                       (log (1+ v))))
             histogram)
    (maphash #'(lambda (k v)
                 (setf (gethash k histogram)
                       (* v (gethash k idf 1))))  ; 1 or N makes little difference
             histogram)
    histogram))

;; non-transformed tf - used for test document tf
(defun tf-plain (vec)
  "Compute term frequencies for VEC, where VEC is, for example, the
  tokenized text of a document."
  (let ((histogram (make-hash-table :test #'equal)))
    (dolist (feature vec)
      (incf (gethash feature histogram 0)))
    histogram))

(defun idf (vecs)
  "Compute inverse document frequencies for VECS, where VECS is a list
  of feature vectors (as returned by osb2vec)."
  (let ((histogram (make-hash-table :test #'equal))
        (doc-inc-map (make-hash-table :test #'equal))
        (N 0))
    (dolist (vec vecs)
      (setq doc-inc-map (make-hash-table :test #'equal))
      (dolist (feature vec)
        (incf N)
        (if (null (gethash feature doc-inc-map))
            (progn
              (incf (gethash feature histogram 0))
              (setf (gethash feature doc-inc-map) t)))))
    (maphash #'(lambda (k v)
                 (setf (gethash k histogram)
                       (log (/ N v))))
             histogram)
    histogram))

(defun train-naive-bayes (training-data)
  "TRAINING-DATA is a list of (class . filename) conses."
  (let ((classes '())
        (class-histogram (make-hash-table :test #'equal))
        (class-features (make-hash-table :test #'equal))
        (vocabulary (make-hash-table :test #'equal))
        (vocablen 0)
        (class-doc-tfs (make-hash-table :test #'equal))
        (class-tf (make-hash-table :test #'equal))
        (idf (make-hash-table :test #'equal))
        (class-tf-sums (make-hash-table :test #'equal))
        (class-priors (make-hash-table :test #'equal))
        (class-likelihoods (make-hash-table :test #'equal))
        ;; (class-likelihood-sums (make-hash-table :test #'equal))
        (classifier-fun))
    (dolist (doc training-data)
      (let ((class (car doc))
            (features (cdr doc)))
        (incf (gethash class class-histogram 0))
        (setf (gethash class class-features)
              (cons features
                    (gethash class class-features '())))))

    (setf classes (keys class-features))

    ;; compute idf
    (setf idf (idf (mapcar #'(lambda (text-list)
                               (apply #'concatenate 'list text-list))
                           (vals class-features))))

    ;; compute tfs - each resultant value is a list of hash tables of tfs
    (maphash #'(lambda (k v)
                 (setf (gethash k class-doc-tfs)
                       (mapcar #'(lambda (v) (tf-train v :idf idf))
                               v)))
             class-features)

    ;; compute log-priors
    ;; priors don't really make any difference and can be assumed to
    ;; be uniform if necessary
    (maphash #'(lambda (k v)
                 (setf (gethash k class-priors)
                       (log (/ v (length training-data)))))
             class-histogram)

    ;; compute global complementary tf table for each class
    ;; for each class...
    (mapcar #'(lambda (class)
                ;; initialize its tf table
                (setf (gethash class class-tf)
                      (make-hash-table :test #'equal))
                ;; for every other class...
                (maphash #'(lambda (other-class doc-tf-list)
                             ;; if it's not the same class...
                             (if (not (equal class other-class))
                                 ;; then, for the tf hash table of each document...
                                 (mapcar #'(lambda (doc-tf)
                                             ;; for every word in the document...
                                             (maphash #'(lambda (word f)
                                                          ;; increment the overall class tf table
                                                          ;; by its frequency
                                                          (incf (gethash word (gethash class class-tf) 0)
                                                                f))
                                                      doc-tf)
                                             ;; (incf (gethash word (gethash class class-tf) 0))
                                             ;; (incf denom (1+ f))
                                             )
                                         doc-tf-list)))
                         class-doc-tfs))
            classes)

    ;; compute vocabulary (global tf table)
    ;; for each class...
    (maphash #'(lambda (k v)
                 (declare (ignore k))
                 ;; for every word in its tf table...
                 (maphash #'(lambda (word f)
                              ;; increment the global tf table by its frequency
                              (incf (gethash word vocabulary 0) f))
                          v))
             class-tf)
    (setf vocablen (length (keys vocabulary)))

    ;; compute complementary class tf sums
    (mapcar #'(lambda (class)
                (setf (gethash class class-tf-sums)
                      (reduce #'+ (vals (gethash class class-tf)))))
            classes)

    ;; compute log-likelihoods
    ;; for each class...
    (mapcar #'(lambda (class)
                ;; initialize its likelihood table
                (setf (gethash class class-likelihoods)
                      (make-hash-table :test #'equal))
                ;; for each word in its complementary tf table
                (maphash #'(lambda (word f)
                             (setf (gethash word (gethash class class-likelihoods))
                                   (log (/ (1+ f)
                                           (+ (gethash class class-tf-sums)
                                              vocablen)))))
                         (gethash class class-tf)))
            classes)

    ;;; decreases accuracy
    ;; ;; compute class likelihood sums
    ;; (mapcar #'(lambda (class)
    ;;             (setf (gethash class class-likelihood-sums)
    ;;                   (reduce #'+ (vals (gethash class class-likelihoods)))))
    ;;         classes)

    ;; ;; normalize likelihoods
    ;; (maphash #'(lambda (class likelihood-table)
    ;;              ;; for each word in its complementary tf table
    ;;             (maphash #'(lambda (word likelihood)
    ;;                          (setf (gethash word likelihood-table)
    ;;                                (/ likelihood (gethash class class-likelihood-sums))))
    ;;                      likelihood-table))
    ;;         class-likelihoods)

    (setf classifier-fun
          (lambda (features)
            (let ((tf (tf-plain features))
                  (output-class)
                  (min-posterior most-positive-fixnum))

              (mapcar #'(lambda (class)
                          (let ((posterior 0))
                            ;; compute posterior
                            (maphash #'(lambda (word f)
                                         ;; add the log-likelihoods
                                         (incf posterior
                                               (* f
                                                  (gethash word (gethash class class-likelihoods)
                                                           (log (/ 1.0
                                                                   (+ (gethash class class-tf-sums)
                                                                      vocablen)))))))
                                     tf)
                            ;; add the log-prior, though this is fairly useless
                            (incf posterior (gethash class class-priors))

                            ;; find min complementary posterior and most probable class
                            ;; (format t "Class ~A has p ~A~%." class posterior)
                            (if (< posterior min-posterior)
                                (setf min-posterior posterior
                                      output-class class))))
                      classes)
              output-class)))

    classifier-fun))

(defun train-naive-bayes-from-files (list-file
                                     &key (tokenize-fun #'default-tokenize-fun)
                                       (feature-fun #'bag-of-words))
  "Train CLASSIFIER on data from LIST-FILE using FEATURE-FUN to
  generate a feature vector for each document and TOKENIZE-FUN to
  tokenize each document.

  FEATURE-FUN takes a list of words as an input and returns a list of
  features. The default is BAG-OF-WORDS, which returns the list of
  words unmodified.

  LIST-FILE should be of the format:
  /path/to/file1 class-of-file1
  /path/to/file2 class-of-file2
  ...
  /path/to/filen class-of-filen

  The file paths should not contain spaces."
  (let ((training-data '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line))
        (let* ((split-line (split-sequence:split-sequence #\Space line :remove-empty-subseqs t))
               (filename (car split-line))
               (class (cadr split-line)))
          (with-open-file (stream filename)
            (let ((text (make-string (file-length stream))))
              (read-sequence text stream)

              (push (cons class (funcall feature-fun (funcall tokenize-fun text)))
                    training-data))))))
    (train-naive-bayes training-data)))

(defun classify-naive-bayes (text classifier-fun
                             &key (tokenize-fun #'default-tokenize-fun)
                               (feature-fun #'bag-of-words))
  "Classify TEXT using the Naive Bayes classifier CLASSIFIER and using
  FEATURE-FUN to generate a feature vector for the document and
  TOKENIZE-FUN to tokenize the document. FEATURE-FUN and TOKENIZE-FUN
  should be the same functions that were used to process training data
  for CLASSIFIER.

  FEATURE-FUN takes a list of words as an input and returns a list of
  features. The default is BAG-OF-WORDS, which returns the list of
  words unmodified.

  TOKENIZE-FUN takes a string as input and returns a list of
  tokens. The default is to use the langutils tokenizer and
  split-sequence with empty subseq removal."
  (let ((features
          (funcall feature-fun (funcall tokenize-fun text))))
    (funcall classifier-fun features)))

(defun classify-file-naive-bayes (filename classifier-fun
                                  &key (tokenize-fun #'default-tokenize-fun)
                                    (feature-fun #'bag-of-words))
  "Classify text from FILENAME. FEATURE-FUN and TOKENIZE-FUN are as in
CLASSIFY-NAIVE-BAYES."
  (with-open-file (stream filename)
    (let ((text (make-string (file-length stream))))
      (read-sequence text stream)
      (classify-naive-bayes text classifier-fun
                            :tokenize-fun tokenize-fun
                            :feature-fun feature-fun))))

(defun classify-files-naive-bayes (list-file classifier-fun
                                   &key (tokenize-fun #'default-tokenize-fun)
                                     (feature-fun #'bag-of-words)
                                     outfile)
  "Classify data from LIST-FILE using CLASSIFIER. FEATURE-FUN and
  TOKENIZE-FUN are as in CLASSIFY-NAIVE-BAYES.

  LIST-FILE should be of the format:
  /path/to/file1 class-of-file1
  /path/to/file2 class-of-file2
  ...
  /path/to/filen class-of-filen

  The file paths should not contain spaces."
  (let ((results '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line))
        (let* ((split-line (split-sequence:split-sequence #\Space line))
               (filename (car split-line))
               (class (cadr split-line)))
          (setq results (acons filename (classify-file-naive-bayes filename
                                                                   classifier-fun
                                                                   :tokenize-fun tokenize-fun
                                                                   :feature-fun feature-fun)
                               results))
          ;; (format t "~A ~A~%" filename (cdar results))
          ))
      (if outfile
          (with-open-file (out outfile :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)
            (dolist (doc (reverse results))
              (format out "~A ~A~%" (car doc) (cdr doc))))))))
