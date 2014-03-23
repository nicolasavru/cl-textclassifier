;;;; cl-textclassifier.lisp

(in-package #:cl-textclassifier)

;;; "cl-textclassifier" goes here. Hacks and glory await!

(defconstant hashlen 32
  "Number of bits of the hash of a feature to use. Also the
  dimensionality for the hyperspace.")

(defvar *hyperspace* '())

(defun bag-of-words (words)
  "Generate the bag of words features of WORDS. This is WORDS itself."
  words
  )

(defun osb (words N)
  "Generate the Orthogonal Sparse Bigram (OSB) features of
  words. Given a list of words (w1 w2 ... wn), its OSB representation
  is ((w1 w2) (w1 w3) ... (w1 wn))"
  (let ((w1 (car words)))
    (mapcar #'(lambda (w) (cons w1 w)) (cdr words))))

;; (defun osb (words N)
;;   "Generate the Orthogonal Sparse Bigram (OSB) features of
;;   words. Given a list of words (w1 w2 ... wn), its OSB representation
;;   is ((w1 w2) (w1 w3) ... (w1 wn))"
;;   (let* ((w1 (car words))
;;         (l (length words))
;;         (arr (make-array l :initial-contents words))
;;         (acc '()))
;;     (dotimes (i (- l N))
;;       (dotimes (j N)
;;         (push (cons (aref arr i) (aref arr (+ i j 1))) acc)
;;         ))
;;     acc))
;;     ;; (mapcar #'(lambda (w) (cons w1 w)) (cdr words))))

(defun features2vec (features)
  "Convert a list of features to a vector in the hyperspace, where
  each vector is represented sparesely as a list of indices
  representing non-zero dimensions. The coefficient in each non-zero
  dimension is assumed to be one. In case case of multiple features
  mapping to the same point, a non-one (and non-zero) dimension is
  represented by two (or more) copies of the vector with unity
  coefficients."
  (mapcar #'(lambda (f) (mod (sxhash f) (expt 2 hashlen))) features))

(defun tf (vecs)
  "Compute term frequencies for VECS, where VECS is a list of feature
  vectors (as returned by osb2vec)."
  (let ((histogram (make-hash-table :test #'equal)))
    (dolist (vec vecs)
      (dolist (feature vec)
        (incf (gethash feature histogram 0))))
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

(defun tf-idf (vecs)
  (let* ((tf-idf-table (make-hash-table :test #'equal))
         (tf-table (tf vecs))
         (idf-table (idf vecs)))
    (maphash #'(lambda (k v)
                (setf (gethash k tf-idf-table)
                      (* v (gethash k idf-table))))
             tf-table)
    tf-idf-table))

(defun train-naive-bayes (docs)
  "DOCS is a list of (class . filename) conses."
  (let ((class-histogram (make-hash-table :test #'equal))
        (class-texts (make-hash-table :test #'equal))
        (vocabulary '())
        (class-feature-vecs (make-hash-table :test #'equal))
        (class-tf (make-hash-table :test #'equal))
        (idf (make-hash-table :test #'equal))
        (class-tf-idf (make-hash-table :test #'equal))
        (class-priors (make-hash-table :test #'equal))
        (class-likelihood-fun)
        (N 0))
    (dolist (doc docs)
      (let ((class (car doc))
            (filename (cdr doc)))
        (incf (gethash class class-histogram 0))
        (with-open-file (stream filename)
          (let ((text (make-string (file-length stream))))
            (read-sequence text stream)
            (multiple-value-bind (success-p idx tokenized-string)
                ;; (declare (ignore success-p))
                (langutils:tokenize-string text)
              (setf (gethash class class-texts)
                  (cons tokenized-string (gethash class class-texts '()))))))))

    ;; compute idf
    (setf idf
          (idf (mapcar #'(lambda (text-list)
                           (apply #'concatenate 'list
                                  (mapcar #'(lambda (text)
                                              (split-sequence:split-sequence
                                               #\Space text
                                               :remove-empty-subseqs t))
                                          text-list)))
                       (vals class-texts))))
    ;; (maphash #'print-hash-entry idf)
    ;; (format t "the ~d~%" (gethash "the" idf))

    ;; (maphash #'(lambda (k v)
    ;;              (setf (gethash k class-idf)
    ;;                    (idf (list (split-sequence:split-sequence #\Space v)))))
    ;;          class-texts)

    ;; concatenate texts for each class
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (setf (gethash k class-texts)
                       (apply #'concatenate 'string (gethash k class-texts))))
             class-texts)

    ;; create vocabulary of all unique words
    (setf vocabulary (remove-duplicates
                      (split-sequence:split-sequence #\Space
                                                     (apply #'concatenate 'string (vals class-texts)))))

    ;; compute log-priors
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (setf (gethash k class-priors)
                       (log (/ (gethash k class-histogram) (length docs)))))
             class-histogram)

    ;; compute tf
    (maphash #'(lambda (k v)
                 (setf (gethash k class-tf)
                       (tf (list (split-sequence:split-sequence #\Space v)))))
             class-texts)
    ;; (maphash #'print-hash-entry
    ;;          (gethash "Str" class-tf))

    ;; compute text length
    (maphash #'(lambda (k v)
                 (incf N
                       (length (split-sequence:split-sequence #\Space v))))
             class-texts)
    (describe N)

    ;; compute tf-idf
    (maphash #'(lambda (k v)
                 (setf (gethash k class-tf-idf)
                       (tf-idf (list (list v)))))
             class-texts)


    ;; compute log-likelihoods with Laplace smoothing
    (setf class-likelihood-fun
          (lambda (word class)
            ;; (format t "word ~s has f ~d~%" word
            ;;         (gethash word (gethash class class-tf) 0))
            (* (log (gethash word idf N))
               (- (log (/ (+ (gethash word
                                      (gethash class class-tf)
                                      0)
                             1)
                          (+ (length (gethash class class-texts))
                             (length vocabulary))))
                  ;; 0
                  ;; CNB
                  (let ((nci-bar 0)
                        (nc-bar 0))
                    (maphash #'(lambda (k v)
                                 (if (not (equal k class))
                                     (progn
                                       (incf nci-bar
                                             (gethash word v 0))
                                       (incf nc-bar
                                             (length (gethash class class-texts))))))
                             class-tf)
                    (log (/ (+ nci-bar 1)
                            (+ nc-bar (length vocabulary))))
                    )
                  ))))
    ;; (maphash #'(lambda (k v)
    ;;              (setf (gethash k class-likelihoods)
    ;;                    (log (/ (+ (gethash k class-tf) 1)
    ;;                            (+ (length (gethash k class-texts))
    ;;                               (length vocabulary))))))
    ;;          class-tf)

    (values t class-priors class-likelihood-fun)))

(defun train-naive-bayes-from-files (list-file)
  "Train CLASSIFIER on data from LIST-FILE. LIST-FILE should be of the
  format:
  /path/to/file1 class-of-file1
  /path/to/file2 class-of-file2
  ...
  /path/to/filen class-of-filen"
  (let ((docs '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line))
        (let* ((split-line (split-sequence:split-sequence #\Space line :remove-empty-subseqs t))
               (filename (car split-line))
               (class (cadr split-line)))
          (push (cons class filename) docs))))
    (train-naive-bayes docs)))

(defun classify-naive-bayes (text class-priors class-likelihood-fun)
  (multiple-value-bind (success-p idx tokenized-string)
                ;; (declare (ignore success-p))
                (langutils:tokenize-string text)
    (let* ((words (split-sequence:split-sequence #\Space tokenized-string :remove-empty-subseqs t))
           (tf (tf (list words)))
           (class-posteriors (make-hash-table :test #'equal))
           (output-class)
           (max-posterior most-negative-fixnum))

      (maphash #'(lambda (class prior)
                   ;; (format t "testing class ~s~%" class)
                   (let ((likelihood 0)
                         (posterior))
                     ;; compute posterior
                     (maphash #'(lambda (word f)
                                  ;; (format t "word ~s has f ~d~%" word f)
                                  (incf likelihood
                                        (* (log (+ 1 f))
                                           (funcall class-likelihood-fun word class))))
                              tf)
                     (setf posterior
                           ;; (+ (gethash class class-priors) likelihood))
                           likelihood)

                     ;; find max posterior and most probable class
                     (format t "class ~s has p ~d~%" class posterior)
                     (if (> posterior max-posterior)
                         (setf max-posterior posterior
                               output-class class))))
               class-priors)
      output-class)))

(defun classify-file-naive-bayes (filename class-priors class-likelihood-fun)
  "Classify text from FILENAME."
  ;; (format t "classifying ~s~%" filename)
  (with-open-file (stream filename)
    (let ((seq (make-string (file-length stream))))
      (read-sequence seq stream)
      (classify-naive-bayes seq
                            class-priors
                            class-likelihood-fun))))

(defun classify-files-naive-bayes (list-file class-priors
                                   class-likelihood-fun &key outfile)
  (format t "outfile: ~s~%" outfile)
  (let ((results '()))
    (with-open-file (stream list-file)
      (do ((line (read-line stream nil)
                 (read-line stream nil)))
          ((null line))
        (let* ((split-line (split-sequence:split-sequence #\Space line))
               (filename (car split-line))
               (class (cadr split-line)))
          (setq results (acons filename (classify-file-naive-bayes filename
                                                                   class-priors
                                                                   class-likelihood-fun) results))
          (format t "~S: ~S~%" filename (classify-file-naive-bayes filename
                                                                   class-priors
                                                                   class-likelihood-fun))
          ))
      (if outfile
          (with-open-file (out outfile :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)
            (dolist (doc (reverse results))
              (format out "~S ~S~%" (car doc) (cdr doc))))))
    ))


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
