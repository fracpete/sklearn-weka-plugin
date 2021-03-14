# scikit-weka
Makes Weka algorithms available in scikit-learn.

It makes use of the [python-weka-wrapper3](https://github.com/fracpete/python-weka-wrapper3) 
library for handling the Java Virtual Machine.


## Functionality

* Available

  * Classifiers (classification/regression)

* Planned

  * Clusterers
  * Data generators
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
  
* install the library

  * from source

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
