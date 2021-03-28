Installation
============

Make sure you have a Java Development Kit (JDK) 8 or later (recommended is 11) installed.

Install the *python-weka-wrapper3* library into a virtual environment; see full instructions here:

  `fracpete.github.io/python-weka-wrapper3/install.html <https://fracpete.github.io/python-weka-wrapper3/install.html>`__

Install the *sklearn-weka-plugin* library itself into the same virtual environment:

* latest release from PyPI

  .. code-block:: bash

     ./venv/bin/pip install sklearn-weka-plugin

* from local source (from within cloned directory)

  .. code-block:: bash

     ./venv/bin/pip install .

* from Github repository

  .. code-block:: bash

     ./venv/bin/pip install git+https://github.com/fracpete/sklearn-weka-plugin.git
