(ql:quickload "cl-textclassifier")
(in-package cl-textclassifier)

(require :sb-sprof)
;; (require :sb-profile)
(declaim (optimize speed))

(let ((labels-fname)
      (fname)
      (outfile))
  (format *query-io* "Enter the input file: ")
  (force-output *query-io*)
  (setf labels-fname (read-line *query-io*))

  (format *query-io* "Enter the output file: ")
  (force-output *query-io*)
  (setf outfile (read-line *query-io*))

  ;; (setq *hyperspace* (classify-files labels-fname
  ;;                                 :classifier *hyperspace*
  ;;                                 :feature-func #'(lambda (words)
  ;;                                                   (osb words 5))))
  ;; (print *hyperspace*)

  (multiple-value-bind (success class-priors class-likelihood-fun)
      (train-naive-bayes-from-files labels-fname)
    (declare (ignore success))

    ;; (format *query-io* "Enter the file to classify: ")
    ;; (force-output *query-io*)
    ;; (setf fname (read-line *query-io*))
    ;; (print (length (car *hyperspace*)))

    ;; (classify-files "./corpus1_test.labels"
    ;;                 :classifier *hyperspace*
    ;;                 :feature-func #'(lambda (words)
    ;;                                   (osb words 5))
    ;;                 :outfile outfile)

    (classify-files-naive-bayes "./corpus1_test.labels"
                                class-priors
                                class-likelihood-fun
                                :outfile outfile))
  )
