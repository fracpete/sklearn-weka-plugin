Changelog
=========

0.0.8 (2024-04-08)
------------------

- using `scikit-learn` instead of deprecated `sklearn` dependency for scikit-learn
  (https://github.com/fracpete/sklearn-weka-plugin/pull/10)


0.0.7 (2023-07-07)
------------------

- `WekaEstimator` (module `sklweka.classifiers`) now has a custom `score` method that
  distinguishes between classification and regression to return the correct score.
- renamed `data` to `X` and `targets` to `y`, since some sklearn schemes use named arguments
- added dummy argument `sample_weight=None` to `fit`, `score` and `fit_predict` methods
- fixed: when supplying Classifier or JBObject instead of classname/options, classname/options
  now get determined automatically
- method `to_instance` (module: `sklweka.dataset`) now performs correct missing value check
- method `to_nominal_labels` (module: `sklweka.dataset`) generates nicer labels now


0.0.6 (2022-04-26)
------------------

- `WekaEstimator` (module `sklweka.classifiers`) and `WekaCluster` (module `sklweka.clusters`)
  now allow specifying how many labels a particular nominal attribute or class attribute has
  (to avoid error message like `Cannot handle unary class attribute!` if there is only one
  label present in a particular split)


0.0.5 (2022-04-01)
------------------

- the `to_nominal_attributes` method in the `sklearn.dataset` module requires now the
  `indices` parameter (incorrectly declared as optional); can parse a range string now as well
- fixed the `fit`, `set_params` and `__str__` methods fo the `MakeNominal` transformer
  (module `sklweka.preprocessing`)
- `WekaEstimator` (module `sklweka.classifiers`) and `WekaCluster` (module `sklweka.clusters`)
  now allow specifying which attributes to turn into nominal ones, which avoids having
  to manually convert the data (either as list with 0-based indices or range string with 1-based indices)
- `set_params` methods now ignore empty dictionaries


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

