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

See the example repository for more information:

http://github.com/fracpete/scikit-weka-examples

Direct links:

* [classifiers](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/classifiers.py)
* [clusters](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/clusters.py)
* [preprocessing](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/preprocessing.py)
* [pipeline](https://github.com/fracpete/scikit-weka-examples/blob/main/src/scikitwekaexamples/pipeline.py)
