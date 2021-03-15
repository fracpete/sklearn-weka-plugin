# scikit-weka
Makes [Weka](https://www.cs.waikato.ac.nz/ml/weka/) algorithms available in [scikit-learn](https://scikit-learn.org/).

Built on top of the [python-weka-wrapper3](https://github.com/fracpete/python-weka-wrapper3) 
library, it uses the [javabridge](https://pypi.python.org/pypi/javabridge) library
under the hood for communicating with Weka objects in the Java Virtual Machine.


## Functionality

Currently available:

* Classifiers (classification/regression)
* Clusters
* Filters

Things to be aware of:

* You need to start/stop the JVM in your Python code before you can use Weka.
* Unlikely to work in multi-threaded/process environments (like flask).
* Jupyter Notebooks does not play nice with javabridge, as you might have to restart the kernel in order to be able to restart the JVM (e.g., with additional packages).
* The conversion to Weka data structures involves guesswork, i.e., if targets are to be treated as nominal, you need to convert the numeric values to strings (e.g., using `scikit.weka.dataset.to_nominal_labels`).


## Installation

* create virtual environment

  ```commandline
  virtualenv -p /usr/bin/python3.7 venv
  ```
  
  or:
  
  ```commandline
  python3 -m venv venv
  ```

* install the *python-weka-wrapper3* library, see instructions here:

  http://fracpete.github.io/python-weka-wrapper3/install.html
  
* install the scikit-weka  library itself

  * from local source

    ```commandline
    ./venv/bin/pip install .   
    ```
    
  * from Github repository

    ```commandline
    ./venv/bin/pip install git+https://github.com/fracpete/scikit-weka.git   
    ```

## Examples

Here is a quick example:

```python
import scikit.weka.jvm as jvm
from scikit.weka.dataset import load_arff, to_nominal_labels
from scikit.weka.classifiers import WekaEstimator
from scikit.weka.clusters import WekaCluster
from scikit.weka.preprocessing import WekaTransformer
from sklearn.model_selection import cross_val_score

# start JVM with Weka package support
jvm.start(packages=True)

# regression
X, y, meta = load_arff("/home/fracpete/development/projects/fracpete/scikit-weka-examples/src/scikitwekaexamples/data/bolts.arff", "last")
lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
scores = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
print("Cross-validating LR on bolts (negRMSE)\n", scores)

# classification
X, y, meta = load_arff("/some/where/iris.arff", "last")
y = to_nominal_labels(y)
j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
j48.fit(X, y)
scores = j48.predict(X)
probas = j48.predict_proba(X)
print("\nJ48 on iris\nactual label -> predicted label, probabilities")
for i in range(len(y)):
    print(y[i], "->", scores[i], probas[i])

# clustering
X, y, meta = load_arff("/some/where/iris.arff", "last")
cl = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
clusters = cl.fit_predict(X)
print("\nSimpleKMeans on iris\nclass label -> cluster")
for i in range(len(y)):
    print(y[i], "->", clusters[i])

# preprocessing
X, y, meta = load_arff("/some/where/bolts.arff", "last")
tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize", options=["-unset-class-temporarily"])
X_new, y_new = tr.fit(X, y).transform(X, y)
print("\nStandardize filter")
print("\ntransformed X:\n", X_new)
print("\ntransformed y:\n", y_new)

# stop JVM
jvm.stop()
```


See the example repository for more examples:

http://github.com/fracpete/scikit-weka-examples

Direct links:

* [classifiers](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/classifiers.py)
* [clusters](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/clusters.py)
* [preprocessing](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/preprocessing.py)
* [pipeline](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/pipeline.py)
