(ql:quickload "cl-textclassifier")
(in-package cl-textclassifier)

(let ((trainfile)
      (testfile)
      (outfile))
  (format *query-io* "Enter the training data metafile: ")
  (force-output *query-io*)
  (setf trainfile (read-line *query-io*))

  (format *query-io* "Enter the testing data metafile: ")
  (force-output *query-io*)
  (setf testfile (read-line *query-io*))


  (format *query-io* "Enter the output file: ")
  (force-output *query-io*)
  (setf outfile (read-line *query-io*))

  (let ((classifier-fun (train-naive-bayes-from-files
                         trainfile
                         :feature-fun #'(lambda (words) (osb words 4))
                         ;; :feature-fun #'(lambda (words) (n-grams words 3))
                         )))
    (format t "Done training!~%")

    (classify-files-naive-bayes testfile
                                classifier-fun
                                :outfile outfile
                                :feature-fun #'(lambda (words) (osb words 4))
                                ;; :feature-fun #'(lambda (words) (n-grams words 3))
                                )
    (format t "Done classifying!~%"))
  )


(sb-ext:exit)
