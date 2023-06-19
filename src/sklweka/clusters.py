import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
from sklweka.preprocessing import to_nominal_attributes
from weka.clusterers import Clusterer
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.serialization import deepcopy
from sklweka.dataset import to_instances, to_instance


class WekaCluster(BaseEstimator, OptionHandler, ClusterMixin):
    """
    Wraps a Weka cluster within the scikit-learn framework.
    """

    def __init__(self, jobject=None, cluster=None, classname=None, options=None, nominal_input_vars=None,
                 num_nominal_input_labels=None):
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
        :param num_nominal_input_labels: the dictionary with the number of labels for the nominal input variables (key is 0-based attribute index)
        :type num_nominal_input_labels: dict
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
        self._classname = classname if (classname is not None) else self._cluster.classname
        self._options = options if (options is not None) else self._cluster.options
        self._nominal_input_vars = nominal_input_vars
        self._num_nominal_input_labels = num_nominal_input_labels

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

    def fit(self, X, y=None, sample_weight=None):
        """
        Trains the cluster.

        :param X: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :param y: ignored
        :type y: ndarray
        :param sample_weight: Sample weights. If None, then samples are equally weighted. TODO Currently ignored.
        :type sample_weight: array-like of shape (n_samples,), default=None
        :return: the cluster
        :rtype: WekaCluster
        """
        if self._nominal_input_vars is not None:
            X = to_nominal_attributes(X, self._nominal_input_vars)
        d = to_instances(X, num_nominal_labels=self._num_nominal_input_labels)
        self._cluster.build_clusterer(d)
        self.header_ = d.template_instances(d, 0)
        return self

    def predict(self, X, y=None):
        """
        Predicts cluster labels.

        :param X: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :param y: ignored
        :type y: ndarray
        :return: the cluster labels (of type int)
        :rtype: ndarray
        """
        check_is_fitted(self)
        if self._nominal_input_vars is not None:
            X = to_nominal_attributes(X, self._nominal_input_vars)
        result = []
        for d in X:
            inst = to_instance(self.header_, d)
            result.append(int(self._cluster.cluster_instance(inst)))
        return np.array(result)

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Trains the cluster and returns the cluster labels.

        :param X: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :param y: ignored
        :type y: ndarray
        :param sample_weight: Sample weights. If None, then samples are equally weighted. TODO Currently ignored.
        :type sample_weight: array-like of shape (n_samples,), default=None
        :return: the cluster labels (of type int)
        :rtype: ndarray
        """
        self.fit(X)
        return self.predict(X)

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
        if self._nominal_input_vars is not None:
            result["nominal_input_vars"] = self._nominal_input_vars
        if self._num_nominal_input_labels is not None:
            result["num_nominal_input_labels"] = self._num_nominal_input_labels
        if self._num_nominal_input_labels is not None:
            result["num_nominal_input_labels"] = self._num_nominal_input_labels
        return result

    def set_params(self, **params):
        """
        Sets the options for the cluster, expects 'classname' and 'options'.

        :param params: the parameter dictionary
        :type params: dict
        """
        if len(params) == 0:
            return
        if "classname" not in params:
            raise Exception("Cannot find 'classname' in parameters!")
        if "options" not in params:
            raise Exception("Cannot find 'options' in parameters!")
        self._classname = params["classname"]
        self._options = params["options"]
        self._cluster = Clusterer(classname=self._classname, options=self._options)
        self._nominal_input_vars = None
        if "nominal_input_vars" in params:
            self._nominal_input_vars = params["nominal_input_vars"]
        self._num_nominal_input_labels = None
        if "num_nominal_input_labels" in params:
            self._num_nominal_input_labels = params["num_nominal_input_labels"]

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
        if isinstance(self._nominal_input_vars, str):
            return "WekaCluster(classname='%s', options=%s, nominal_input_vars='%s')" % (self._cluster.classname, str(self._cluster.options), str(self._nominal_input_vars))
        else:
            return "WekaCluster(classname='%s', options=%s, nominal_input_vars=%s)" % (self._cluster.classname, str(self._cluster.options), str(self._nominal_input_vars))
