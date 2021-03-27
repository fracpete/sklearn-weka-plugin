import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from weka.classifiers import Classifier
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.dataset import missing_value
from weka.core.serialization import deepcopy
from sklweka.dataset import to_instances, to_instance


class WekaEstimator(BaseEstimator, OptionHandler, RegressorMixin, ClassifierMixin):
    """
    Wraps a Weka classifier (classifier/regressor) within the scikit-learn framework.
    """

    def __init__(self, jobject=None, classifier=None, classname=None, options=None):
        """
        Initializes the estimator. Can be either instantiated via the following priority of parameters:
        1. JB_Object representing a Java Classifier object
        2. Classifier pww3 wrapper
        3. classname/options

        :param jobject: the JB_Object representing a Weka classifier to use
        :type jobject: JB_Object
        :param classifier: the classifier wrapper to use
        :type classifier: Classifier
        :param classname: the classname of the Weka classifier to instantiate
        :type classname: str
        :param options: the command-line options of the Weka classifier to instantiate
        :type options: list
        """
        if jobject is not None:
            _jobject = jobject
        elif classifier is not None:
            _jobject = classifier.jobject
        elif classname is not None:
            if options is None:
                options = []
            classifier = Classifier(classname=classname, options=options)
            _jobject = classifier.jobject
        else:
            raise Exception("At least Java classname must be provided!")

        if not is_instance_of(_jobject, "weka.classifiers.Classifier"):
            raise Exception("Java object does not implement weka.classifiers.Classifier!")

        super(WekaEstimator, self).__init__(_jobject)
        self._classifier = Classifier(jobject=_jobject)
        self.header_ = None
        # the following references are required for get_params/set_params
        self._classname = classname
        self._options = options

    @property
    def classifier(self):
        """
        Returns the underlying classifier object, if any.

        :return: the classifier object
        :rtype: Classifier
        """
        return self._classifier

    @property
    def header(self):
        """
        Returns the underlying dataset header, if any.

        :return: the dataset structure
        :rtype: Instances
        """
        return self.header_

    def fit(self, data, targets):
        """
        Trains the estimator.

        :param data: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: the class attribute column, array-like of shape (n_samples,)
        :type targets: ndarray
        :return: itself
        :rtype: WekaEstimator
        """
        data, targets = check_X_y(data, y=targets)
        d = to_instances(data, targets)
        self._classifier.build_classifier(d)
        self.header_ = d.template_instances(d, 0)
        return self

    def predict(self, data):
        """
        Performs predictions with the trained classifier.

        :param data: the data matrix to generate predictions for, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :return: the score (or scores)
        :rtype: ndarray
        """
        check_is_fitted(self)
        data = check_array(data)
        result = []
        for d in data:
            inst = to_instance(self.header_, d, missing_value())
            if self.header_.class_attribute.is_nominal:
                result.append(self.header_.class_attribute.value(int(self._classifier.classify_instance(inst))))
            else:
                result.append(self._classifier.classify_instance(inst))
        return np.array(result)

    def predict_proba(self, data):
        """
        Performs predictions and returns class probabilities.

        :param data: the data matrix to generate predictions for, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :return: the probabilities
        """
        check_is_fitted(self)
        data = check_array(data)
        result = []
        for d in data:
            inst = to_instance(self.header_, d, missing_value())
            result.append(self._classifier.distribution_for_instance(inst))
        return np.array(result)

    def get_params(self, deep=True):
        """
        Returns the parameters for this classifier, basically classname and options list.

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
        Sets the options for the classifier, expects 'classname' and 'options'.

        :param params: the parameter dictionary
        :type params: dict
        """
        if "classname" not in params:
            raise Exception("Cannot find 'classname' in parameters!")
        if "options" not in params:
            raise Exception("Cannot find 'options' in parameters!")
        self._classname = params["classname"]
        self._options = params["options"]
        self._classifier = Classifier(classname=self._classname, options=self._options)

    def __str__(self):
        """
        For printing the model.

        :return: the model representation, if any
        :rtype: str
        """
        if self._classifier is None:
            return self._classname + ": No model built yet"
        else:
            return str(self._classifier)

    def __copy__(self):
        """
        Creates a deep copy of itself.

        :return: the copy
        :rtype: WekaEstimator
        """
        result = WekaEstimator(jobject=deepcopy(self.jobject))
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
        return "WekaEstimator(classname='%s', options=%s)" % (self._classifier.classname, str(self._classifier.options))
