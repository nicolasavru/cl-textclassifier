# cl-textclassifier

cl-textclassifier is a text classifier implementation in Common
Lisp. A Naive Bayes classifier as described in [1] is used. However,
some modifications from [1] proved to be detrimental in testing and
are disabled.

References:
[1]  http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

## Usage

### Installation

1. Install quicklisp. $ represents a bash shell and * an Lisp REPL.
```$ wget http://beta.quicklisp.org/quicklisp.lisp && sbcl --load quicklisp.lisp
   * (quicklisp-quickstart:install)
   * (ql:add-to-init-file)
   * (sb-ext:exit)
```


2. Configure ASDF2:
```$ mkdir -p ~/.config/common-lisp/source-registry.conf.d/```

~/.config/common-lisp/source-registry.conf.d/projects.conf should
contain the following contents:
```(:tree (:home "Documents/lisp/"))```

"Documents/lisp/" is a path relative to your home directory that ASDF
will search for lisp projects in.

3. Run cl-textclassifier:
```$ sbcl --load run.lisp```

For the sake of your sanity, I suggest running sbcl through rlwrap:
```$ rlwrap sbcl```

### Runtime

