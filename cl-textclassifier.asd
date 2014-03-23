;;;; cl-nn.asd

(asdf:defsystem #:cl-textclassifier
  :serial t
  :description "A text classifier implementated in Common Lisp."
  :author "Nicolas Avrutin <nicolasavru@gmail.com>"
  :depends-on (#:langutils #:split-sequence)
  :components ((:file "package")
               (:file "cl-textclassifier")))

