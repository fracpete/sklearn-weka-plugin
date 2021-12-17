Changelog
=========

0.0.5 (????-??-??)
------------------

- ...


0.0.4 (2021-12-17)
------------------

- fixed sorting of labels in `to_instances` method in module `sklweka.dataset`
- redoing `X` when no class present in `load_arff` method (module `sklweka.dataset`)
- added `load_dataset` method in module `sklweka.dataset` that uses Weka to load the
  data before converting it into sklearn data structures (slower, but more flexible)


0.0.3 (2021-10-26)
------------------

- added support for generating data via Weka's data generators


0.0.2 (2021-04-12)
------------------

- requiring python-weka-wrapper3 version 0.2.1 at least in order to offer pickle support


0.0.1 (2021-03-28)
------------------

- initial release

