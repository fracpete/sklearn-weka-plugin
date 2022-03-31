Examples
========

The following examples are meant to be executed in sequence, as they rely on previous steps,
e.g., on data present.


Start the Java Virtual Machine
------------------------------

*sklearn-weka-plugin* simply imports everything from `weka.core.jvm` into `sklweka.jvm` for convenience:

.. code-block:: python

   import sklweka.jvm as jvm

   # start JVM with Weka package support
   jvm.start(packages=True)


Location of the datasets
------------------------

The following examples assume the datasets to be present in the `data_dir` directory. For instance,
this could be the following directory:

.. code-block:: python

   data_dir = "/my/datasets/"


Loading data
------------

There are two methods available from the `sklweka.dataset` module for loading data:

* `load_arff` - uses the `loadarff` method from the module `scipy.io.arff`
  (cannot handle string attributes). Nominal classes have to be converted using
  the `to_nominal_labels` method.
* `load_dataset` - uses Weka's own data loading functionality before converting it
  into sklearn data structures, i.e., numpy matrices. Bit slower due to data conversion,
  but handles string attributes. Also not limited to ARFF files. The data can be
  either returned with mixed types (not necessary to use the `to_nominal_labels` method
  then) or in Weka's internal, numeric-only data format.


Regression
----------

Regression algorithms are available through the `sklweka.classifiers.WekaEstimator` wrapper:

.. code-block:: python

   from sklweka.dataset import load_arff
   from sklweka.classifiers import WekaEstimator
   from sklearn.model_selection import cross_val_score

   X, y, meta = load_arff(data_dir + "/bolts.arff", class_index="last")
   lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
   scores = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
   print("Cross-validating LR on bolts (negRMSE)\n", scores)


Classification
--------------

Classification algorithms are also available through the `sklweka.classifiers.WekaEstimator` wrapper:

.. code-block:: python

   from sklweka.dataset import load_arff, to_nominal_labels, load_dataset
   from sklweka.classifiers import WekaEstimator
   from sklearn.model_selection import cross_val_score

   # using the load_arff method:
   X, y, meta = load_arff(data_dir + "/iris.arff", class_index="last")
   y = to_nominal_labels(y)
   # using the load_dataset method:
   # X, y = load_dataset(data_dir + "/iris.arff", class_index="last")

   j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
   j48.fit(X, y)
   scores = j48.predict(X)
   probas = j48.predict_proba(X)
   print("\nJ48 on iris\nactual label -> predicted label, probabilities")
   for i in range(len(y)):
       print(y[i], "->", scores[i], probas[i])

Alternatively to manually converting labels and nominal attributes, you can use
the `nominal_input_vars` (list of 0-based indices or range string with 1-based
indices) and `nominal_output_var` (bool) parameters in the constructor:

.. code-block:: python

   from sklweka.dataset import load_arff
   from sklweka.classifiers import WekaEstimator
   from sklearn.model_selection import cross_val_score

   # using the load_arff method:
   X, y, meta = load_arff(data_dir + "/vote.arff", class_index="last")

   j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"],
                       nominal_input_vars="first-last", nominal_output_var=True)
   j48.fit(X, y)
   scores = j48.predict(X)
   probas = j48.predict_proba(X)
   print("\nJ48 on iris\nactual label -> predicted label, probabilities")
   for i in range(len(y)):
       print(y[i], "->", scores[i], probas[i])


Clustering
----------

Clustering algorithms are available through the `sklweka.clusters.WekaCluster` wrapper:

.. code-block:: python

   from sklweka.dataset import load_arff
   from sklweka.clusters import WekaCluster

   X, y, meta = load_arff(data_dir + "/iris.arff", class_index="last")
   cl = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
   clusters = cl.fit_predict(X)
   print("\nSimpleKMeans on iris\nclass label -> cluster")
   for i in range(len(y)):
       print(y[i], "->", clusters[i])

Alternatively to manually converting labels and nominal attributes, you can use
the `nominal_input_vars` (list of 0-based indices or range string with 1-based
indices) parameter in the constructor:

.. code-block:: python

   from sklweka.dataset import load_arff
   from sklweka.clusters import WekaCluster

   X, y, meta = load_arff(data_dir + "/vote.arff", class_index="last")
   cl = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"],
                    nominal_input_vars=[x for x in range(X.shape[1])])
   clusters = cl.fit_predict(X)
   print("\nSimpleKMeans on iris\nclass label -> cluster")
   for i in range(len(y)):
       print(y[i], "->", clusters[i])


Preprocessing
-------------

Weka filters can be applied by using the `sklweka.preprocessing.WekaTransformer` wrapper:

.. code-block:: python

   from sklweka.dataset import load_arff
   from sklweka.preprocessing import WekaTransformer

   X, y, meta = load_arff(data_dir + "/bolts.arff", class_index="last")
   tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize", options=["-unset-class-temporarily"])
   X_new, y_new = tr.fit(X, y).transform(X, y)
   print("\nStandardize filter")
   print("\ntransformed X:\n", X_new)
   print("\ntransformed y:\n", y_new)


Data generators
---------------

Weka's data generators can be used for generating numpy arrays as well:

.. code-block:: python

   from sklweka.datagenerators import DataGenerator, generate_data

   gen = DataGenerator(
       classname="weka.datagenerators.classifiers.classification.BayesNet",
       options=["-S", "2", "-n", "10", "-C", "10"])
   X, y, X_names, y_name = generate_data(gen, att_names=True)
   print("X:", X_names)
   print(X)
   print("y:", y_name)
   print(y)


Stop the Java Virtual Machine
-----------------------------

At end of your Python script, stop the JVM as follows:

.. code-block:: python

   jvm.stop()


**NB:** The JVM cannot be restarted within the same Python process, a drawback of the underlying
*javabridge* library.


Additional examples
-------------------

More examples can be found at:

`github.com/fracpete/sklearn-weka-plugin-examples <http://github.com/fracpete/sklearn-weka-plugin-examples>`__
