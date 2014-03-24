;;;; cl-textclassifier.lisp

(in-package #:cl-textclassifier)

;;; "cl-textclassifier" goes here. Hacks and glory await!

(defconstant hashlen 32
  "Number of bits of the hash of a feature to use. Also the
  dimensionality for the hyperspace.")

(defun bag-of-words (words)
  "Generate the bag of words features of WORDS. This is WORDS itself."
  words
  )

;; (defun n-grams (words N)
;;   (let* ((w1 (car words))
;;         (l (length words))
;;         (arr (make-array l :initial-contents words))
;;         (acc '()))
;;     (dotimes (i (- l N))
;;       (dotimes (j N)
;;         (push (cons (aref arr i) (aref arr (+ i j 1))) acc)
;;         ))
;;     acc))

(defun osb (words)
  "Generate the Orthogonal Sparse Bigram (OSB) features of
  words. Given a list of words (w1 w2 ... wn), its OSB representation
  is ((w1 w2) (w1 w3) ... (w1 wn))"
  (let ((w1 (car words)))
    (mapcar #'(lambda (w) (sxhash (cons w1 w))) (cdr words))))

(defun osb-sliding (words N)
  "Generate the Orthogonal Sparse Bigram (OSB) features of
  words. Given a list of words (w1 w2 ... wn), its OSB representation
  is ((w1 w2) (w1 w3) ... (w1 wn))"
  (let* ((w1 (car words))
        (l (length words))
        (arr (make-array l :initial-contents words))
        (acc '()))
    (dotimes (i (- l N))
      (dotimes (j N)
        (push (cons (aref arr i) (aref arr (+ i j 1))) acc)
        ))
    acc))
    ;; (mapcar #'(lambda (w) (cons w1 w)) (cdr words))))

(defun features2vec (features)
  "Convert a list of features to a vector in the hyperspace, where
  each vector is represented sparesely as a list of indices
  representing non-zero dimensions. The coefficient in each non-zero
  dimension is assumed to be one. In case case of multiple features
  mapping to the same point, a non-one (and non-zero) dimension is
  represented by two (or more) copies of the vector with unity
  coefficients."
  (mapcar #'(lambda (f) (mod (sxhash f) (expt 2 hashlen))) features))

;; (defun tf (vecs)
;;   "Compute term frequencies for VECS, where VECS is a list of feature
;;   vectors (as returned by osb2vec)."
;;   (let ((histogram (make-hash-table :test #'equal)))
;;     (dolist (vec vecs)
;;       (dolist (feature vec)
;;         (incf (gethash feature histogram 0))))
;;     histogram))

(defun tf (vec &key idf)
  "Compute term frequencies for VEC, where VEC is a list of features."
  (let ((histogram (make-hash-table :test #'equal))
        (denom 0))
    (dolist (feature vec)
      (incf (gethash feature histogram 0)))
    (maphash #'(lambda (k v)
                 (setf (gethash k histogram)
                       (log (1+ v)))
                 (incf denom (expt (gethash k histogram) 2)))
             histogram)
    (setf denom (sqrt denom))
    (maphash #'(lambda (k v)
                 (setf (gethash k histogram)
                       (* (gethash k idf 1) ; TODO-maybe: change 1 to N
                          (/ v denom))))
             histogram)
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
  "Train a naive bayes classifier on TRAINING-DATA, where
  TRAINING-DATA is a list of (class . text) conses."
  (let ((classes '())
        (class-histogram (make-hash-table :test #'equal))
        (class-features (make-hash-table :test #'equal))
        (vocabulary (make-hash-table :test #'equal))
        (class-feature-vecs (make-hash-table :test #'equal))
        (class-doc-tfs (make-hash-table :test #'equal))
        (class-tf (make-hash-table :test #'equal))
        (idf (make-hash-table :test #'equal))
        (class-tf-idf (make-hash-table :test #'equal))
        (class-tf-sums (make-hash-table :test #'equal))
        (class-priors (make-hash-table :test #'equal))
        (class-likelihoods (make-hash-table :test #'equal))
        (class-likelihood-sums (make-hash-table :test #'equal))
        (class-likelihood-fun)
        (N 0))
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
    ;; (maphash #'print-hash-entry idf)
    ;; (format t "the ~d~%" (gethash "the" idf))

    ;; compute tfs - each resultant value is a list of hash tables of tfs
    (maphash #'(lambda (k v)
                 (setf (gethash k class-doc-tfs)
                       (mapcar #'(lambda (v) (tf v :idf idf))
                               v)))
             class-features)

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
                 ;; for every word in its tf table...
                 (maphash #'(lambda (word f)
                              ;; increment the global tf table by its frequency
                              (incf (gethash word vocabulary 0) f))
                          v))
             class-tf)

    ;; compute complementary class tf sums
    (mapcar #'(lambda (class)
                (setf (gethash class class-tf-sums)
                      (reduce #'+ (vals (gethash class class-tf)))))
            classes)

    ;; compute likelihoods
    ;; for each class...
    (mapcar #'(lambda (class)
                ;; initialize its likelihood table
                (setf (gethash class class-likelihoods)
                      (make-hash-table :test #'equal))
                ;; for each word in its complementary tf table
                (maphash #'(lambda (word f)
                             (setf (gethash word (gethash class class-likelihoods))
                                   (log (/ (1+ (gethash word (gethash class class-tf)))
                                           (+ (gethash class class-tf-sums)
                                              (length (keys vocabulary)))))))
                         (gethash class class-tf)))
            classes)

    ;; compute class likelihood sums
    (mapcar #'(lambda (class)
                (setf (gethash class class-likelihood-sums)
                      (reduce #'+ (vals (gethash class class-likelihoods)))))
            classes)

    ;; normalize likelihoods
    (maphash #'(lambda (class likelihood-table)
                 ;; for each word in its complementary tf table
                (maphash #'(lambda (word likelihood)
                             (setf (gethash word likelihood-table)
                                   (/ likelihood (gethash class class-likelihood-sums))))
                         likelihood-table))
            class-likelihoods)


    (setf class-likelihood-fun
          (lambda (word class)
            (gethash word (gethash class class-likelihoods)
                     (log (/ 1.0
                             (+ (gethash class class-tf-sums)
                                (length (keys vocabulary))))))))

    ;; don't bother computing priors
    (maphash #'(lambda (k v)
                 (declare (ignore k v))
                 (setf (gethash k class-priors)
                       1))
             class-histogram)

    (values t class-priors class-likelihood-fun idf)))

(defun train-naive-bayes-from-files (list-file &key (feature-fun #'bag-of-words))
  "Train CLASSIFIER on data from LIST-FILE. LIST-FILE should be of the
  format:
  /path/to/file1 class-of-file1
  /path/to/file2 class-of-file2
  ...
  /path/to/filen class-of-filen

  Assumes the file paths do not contain spaces."
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

              (push (cons class
                          ;; feature-fun
                          ;; (funcall feature-fun
                          (split-sequence:split-sequence #\Space
                                                         ;; tokenization fun
                                                         (multiple-value-bind
                                                               (success-p idx tokenized-string)
                                                             (langutils:tokenize-string text)
                                                           (declare (ignore success-p))
                                                           tokenized-string)
                                                         :remove-empty-subseqs t))
                    ;; )
                    training-data))))))
    (train-naive-bayes training-data)))

(defun classify-naive-bayes (text class-priors class-likelihood-fun idf
                             &key (feature-fun #'bag-of-words))
  (multiple-value-bind (success-p idx tokenized-string)
                ;; (declare (ignore success-p))
                (langutils:tokenize-string text)
    (let* ((words
             ;; (funcall feature-fun
             (split-sequence:split-sequence #\Space tokenized-string :remove-empty-subseqs t))
           ;; )
           (tf (tf words :idf idf))
           (class-posteriors (make-hash-table :test #'equal))
           (output-class)
           (min-posterior most-positive-fixnum))

      (maphash #'(lambda (class prior)
                   (let ((likelihood 0)
                         (posterior))
                     ;; compute posterior
                     (maphash #'(lambda (word f)
                                  (incf likelihood
                                        (* f (funcall class-likelihood-fun word class))))
                              tf)
                     (setf posterior
                           ;; (+ (gethash class class-priors) likelihood))
                           likelihood)

                     ;; find max posterior and most probable class
                     ;; (format t "class ~s has p ~d~%" class posterior)
                     (if (< posterior min-posterior)
                         (setf min-posterior posterior
                               output-class class))))
               class-priors)
      output-class)))

(defun classify-file-naive-bayes (filename class-priors class-likelihood-fun idf
                                  &key (feature-fun #'bag-of-words))
  "Classify text from FILENAME."
  ;; (format t "classifying ~s~%" filename)
  (with-open-file (stream filename)
    (let ((seq (make-string (file-length stream))))
      (read-sequence seq stream)
      (classify-naive-bayes seq class-priors
                            class-likelihood-fun idf
                            :feature-fun feature-fun))))

(defun classify-files-naive-bayes (list-file class-priors
                                   class-likelihood-fun idf
                                   &key (feature-fun #'bag-of-words) outfile)
  (let ((results '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line))
        (let* ((split-line (split-sequence:split-sequence #\Space line))
               (filename (car split-line))
               (class (cadr split-line)))
          (format t "classifying ~a~%" filename)
          (setq results (acons filename (classify-file-naive-bayes filename
                                                                   class-priors
                                                                   class-likelihood-fun idf
                                                                   :feature-fun feature-fun) results))
          ;; (format t "~S: ~S~%" filename (classify-file-naive-bayes filename
          ;;                                                          class-priors
          ;;                                                          class-likelihood-fun idf
          ;;                                                          :feature-fun feature-fun))
          ))
      (if outfile
          (with-open-file (out outfile :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)
            (dolist (doc (reverse results))
              (format out "~S ~S~%" (car doc) (cdr doc))))))))


(defun keys (table)
  (let ((acc '()))
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (push k acc))
             table)
    (nreverse acc)))

(defun vals (table)
  (let ((acc '()))
    (maphash #'(lambda (k v)
                 (declare (ignore k))
                 (push v acc))
             table)
    (nreverse acc)))


(defun learn (class text &key classifier feature-func)
  "Train CLASSIFIER on TEXT, which is in CLASS. Return CLASSIFIER"
  (multiple-value-bind (success-p idx tokenized-string)
      ;; (declare (ignore success-p))
      (langutils:tokenize-string text)
    (let ((words (split-sequence:split-sequence #\Space tokenized-string :remove-empty-subseqs t)))
      (acons class (features2vec (funcall feature-func words)) classifier))))

(defun learn-from-file (class filename &key classifier feature-func)
  "Train CLASSIFIER on text read from FILENAME, which is in CLASS"
  (with-open-file (stream filename)
    (let ((seq (make-string (file-length stream))))
      (read-sequence seq stream)
      (setq classifier (learn class seq
                              :classifier classifier
                              :feature-func feature-func
                              )))))

(defun learn-files (list-file &key classifier feature-func)
  "Train CLASSIFIER on data from LIST-FILE. LIST-FILE should be of the
  format:
  /path/to/file1 class-of-file1
  /path/to/file2 class-of-file2
  ...
  /path/to/filen class-of-filen"
  (with-open-file (stream list-file)
    (do ((line (read-line stream nil)
               (read-line stream nil)))
        ((null line) classifier)
      (let* ((split-line (split-sequence:split-sequence #\Space line))
             (filename (car split-line))
             (class (cadr split-line)))
        (setq classifier (learn-from-file class filename
                                          :classifier classifier
                                          :feature-func feature-func))))))

(defun print-hash-entry (key value)
  (format t "The value associated with the key ~S is ~S~%" key value))

(defun classify (text &key classifier feature-func)
  "Classify TEXT using CLASSIFIER. Return the class to which TEXT is
predicted to belong to."
  (multiple-value-bind (success-p idx tokenized-string)
      ;; (declare (ignore success-p))
      (langutils:tokenize-string text)
    (let* ((words (split-sequence:split-sequence #\Space tokenized-string :remove-empty-subseqs t))
           (new-vec (features2vec (funcall feature-func words)))
           (class-radiance-table (make-hash-table :test #'equal))
           (nfeats 0)
           (max-radiance 0)
           (max-class))
      (dolist (labeled-vec classifier)
        (let* ((label (car labeled-vec))
               (known-vec (cdr labeled-vec))
               (dist (sqrt (length (set-exclusive-or known-vec new-vec))))
               (kandu (length (intersection known-vec new-vec)))
               ;; (dist (/ (+ (expt (/ (length (set-exclusive-or known-vec new-vec)) 2) 2) 1)
               ;;          (+ (* 4 kandu kandu) 1)))
               (radiance (* (/ 1.0 (+ (* dist dist) .000001)) kandu kandu))
               ;; (radiance (/ 1.0  dist))
               )
          (incf nfeats)
          (incf (gethash label class-radiance-table 0) radiance)
        ;; TODO-maybe: normalize radiance by number of features
          ))
      ;; (maphash #'print-hash-entry class-radiance-table)
      (maphash #'(lambda (key value)
                   (if (> value max-radiance)
                       (setq max-radiance value
                             max-class key)))
               class-radiance-table)
      max-class)))

(defun classify-file (filename &key classifier feature-func)
  "Classify text from FILENAME."
  (with-open-file (stream filename)
    (let ((seq (make-string (file-length stream))))
      (read-sequence seq stream)
      (classify seq
                :classifier classifier
                :feature-func feature-func))))

(defun classify-files (list-file &key classifier feature-func outfile)
  (format t "outfile: ~s~%" outfile)
  (let ((results '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line) classifier)
        (let* ((split-line (split-sequence:split-sequence #\Space line))
               (filename (car split-line))
               (class (cadr split-line)))
          (setq results (acons filename (classify-file filename
                                                       :classifier classifier
                                                       :feature-func feature-func) results))
          (format t "~S: ~S~%" filename (classify-file filename
                                                       :classifier classifier
                                                       :feature-func feature-func))
          ))
      (if outfile
          (with-open-file (out outfile :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)
            (dolist (doc (reverse results))
              (format out "~S ~S~%" (car doc) (cdr doc))))))))
