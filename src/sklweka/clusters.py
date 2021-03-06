import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from weka.clusterers import Clusterer
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.serialization import deepcopy
from sklweka.dataset import to_instances, to_instance


class WekaCluster(BaseEstimator, OptionHandler, ClusterMixin):
    """
    Wraps a Weka cluster within the scikit-learn framework.
    """

    def __init__(self, jobject=None, cluster=None, classname=None, options=None):
        """
        Initializes the estimator. Can be either instantiated via the following priority of parameters:
        1. JB_Object representing a Java Clusterer object
        2. Clusterer pww3 wrapper
        3. classname/options

        :param jobject: the JB_Object representing a Weka cluster to use
        :type jobject: JB_Object
        :param cluster: the cluster wrapper to use
        :type cluster: Clusterer
        :param classname: the classname of the Weka cluster to instantiate
        :type classname: str
        :param options: the command-line options of the Weka cluster to instantiate
        :type options: list
        """
        if jobject is not None:
            _jobject = jobject
        elif cluster is not None:
            _jobject = cluster.jobject
        elif classname is not None:
            if options is None:
                options = []
            cluster = Clusterer(classname=classname, options=options)
            _jobject = cluster.jobject
        else:
            raise Exception("At least Java classname must be provided!")

        if not is_instance_of(_jobject, "weka.clusterers.Clusterer"):
            raise Exception("Java object does not implement weka.clusterers.Clusterer!")

        super(WekaCluster, self).__init__(_jobject)
        self._cluster = Clusterer(jobject=_jobject)
        self.header_ = None
        # the following references are required for get_params/set_params
        self._classname = classname
        self._options = options

    @property
    def cluster(self):
        """
        Returns the underlying cluster object, if any.

        :return: the cluster object
        :rtype: Clusterer
        """
        return self._cluster

    @property
    def header(self):
        """
        Returns the underlying dataset header, if any.

        :return: the dataset structure
        :rtype: Instances
        """
        return self.header_

    def fit(self, data, targets=None):
        """
        Trains the cluster.

        :param data: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: ignored
        :type targets: ndarray
        :return: the cluster
        :rtype: WekaCluster
        """
        d = to_instances(data)
        self._cluster.build_clusterer(d)
        self.header_ = d.template_instances(d, 0)
        return self

    def predict(self, data, targets=None):
        """
        Predicts cluster labels.

        :param data: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: ignored
        :type targets: ndarray
        :return: the cluster labels (of type int)
        :rtype: ndarray
        """
        check_is_fitted(self)
        result = []
        for d in data:
            inst = to_instance(self.header_, d)
            result.append(int(self._cluster.cluster_instance(inst)))
        return np.array(result)

    def fit_predict(self, data, targets=None):
        """
        Trains the cluster and returns the cluster labels.

        :param data: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: ignored
        :type targets: ndarray
        :return: the cluster labels (of type int)
        :rtype: ndarray
        """
        self.fit(data)
        return self.predict(data)

    def get_params(self, deep=True):
        """
        Returns the parameters for this cluster, basically classname and options list.

        :param deep: ignored
        :type deep: bool
        :return: the dictionary with options
        :rtype: dict
        """
        result = dict()
        result["classname"] = self._classname
        result["options"] = self._options
        return result

    def set_params(self, **params):
        """
        Sets the options for the cluster, expects 'classname' and 'options'.

        :param params: the parameter dictionary
        :type params: dict
        """
        if "classname" not in params:
            raise Exception("Cannot find 'classname' in parameters!")
        if "options" not in params:
            raise Exception("Cannot find 'options' in parameters!")
        self._classname = params["classname"]
        self._options = params["options"]
        self._cluster = Clusterer(classname=self._classname, options=self._options)

    def __str__(self):
        """
        For printing the model.

        :return: the model representation, if any
        :rtype: str
        """
        if self._cluster is None:
            return self._classname + ": No model built yet"
        else:
            return str(self._cluster)

    def __copy__(self):
        """
        Creates a deep copy of itself.

        :return: the copy
        :rtype: WekaEstimator
        """
        result = WekaCluster(jobject=deepcopy(self.jobject))
        result._classname = self._classname
        result._options = self._options[:]
        return result

    def __repr__(self, N_CHAR_MAX=700):
        """
        Returns a valid Python string using its classname and options.

        :param N_CHAR_MAX: ignored
        :type N_CHAR_MAX: int
        :return: the representation
        :rtype: str
        """
        return "WekaCluster(classname='%s', options=%s)" % (self._cluster.classname, str(self._cluster.options))
