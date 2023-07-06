.. sklearn-weka-plugin documentation master file, created by
   sphinx-quickstart on Sat Apr 12 11:51:06 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Makes `Weka <https://www.cs.waikato.ac.nz/ml/weka/>`__ algorithms available in
`scikit-learn <https://scikit-learn.org/>`__.

Built on top of the `python-weka-wrapper3 <https://github.com/fracpete/python-weka-wrapper3>`__ library,
it uses the `javabridge <https://github.com/fracpete/python-weka-wrapper3>`__ library under the hood for
communicating with Weka objects in the Java Virtual Machine.

Links:

* Looking for code?

  * `Project homepage <https://github.com/fracpete/sklearn-weka-plugin>`__
  * `Example code <https://github.com/fracpete/sklearn-weka-plugin-examples>`__


Requirements
============

The library has the following requirements:

* Python 3 (does not work with Python 2)

  * python-weka-wrapper (>=0.2.5, required)

* OpenJDK 8 or later (11 is recommended)

Contents
========

.. toctree::
   :maxdepth: 2

   install
   examples
   gotchas

API
===

.. toctree::
   :maxdepth: 4

   sklweka
