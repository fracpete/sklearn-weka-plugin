import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, r2_score
from sklweka.preprocessing import to_nominal_attributes, to_nominal_labels
from weka.classifiers import Classifier
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.dataset import missing_value
from weka.core.serialization import deepcopy
from sklweka.dataset import to_instances, to_instance


class WekaEstimator(BaseEstimator, OptionHandler, RegressorMixin, ClassifierMixin):
    """
    Wraps a Weka classifier (classifier/regressor) within the scikit-learn framework.
    """

    def __init__(self, jobject=None, classifier=None, classname=None, options=None,
                 nominal_input_vars=None, nominal_output_var=None,
                 num_nominal_input_labels=None, num_nominal_output_labels=None):
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
        :param nominal_input_vars: the list of 0-based indices of attributes to convert to nominal or range string with 1-based indices
        :type nominal_input_vars: list or str
        :param nominal_output_var: whether to convert the output variable to a nominal one
        :type nominal_output_var: bool
        :param num_nominal_input_labels: the dictionary with the number of labels for the nominal input variables (key is 0-based attribute index)
        :type num_nominal_input_labels: dict
        :param num_nominal_output_labels: the number of labels for the output variable
        :type num_nominal_output_labels: int
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
        self.classes_ = None
        # the following references are required for get_params/set_params
        self._classname = classname if (classname is not None) else self._classifier.classname
        self._options = options if (options is not None) else self._classifier.options
        self._nominal_input_vars = nominal_input_vars
        self._nominal_output_var = nominal_output_var
        self._num_nominal_input_labels = num_nominal_input_labels
        self._num_nominal_output_labels = num_nominal_output_labels

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

    def fit(self, X, y, sample_weight=None):
        """
        Trains the estimator.

        :param X: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :param y: the class attribute column, array-like of shape (n_samples,)
        :type y: ndarray
        :param sample_weight: Sample weights. If None, then samples are equally weighted. TODO Currently ignored.
        :type sample_weight: array-like of shape (n_samples,), default=None
        :return: itself
        :rtype: WekaEstimator
        """
        X, y = check_X_y(X, y=y, dtype=None)
        if self._nominal_input_vars is not None:
            X = to_nominal_attributes(X, self._nominal_input_vars)
        if self._nominal_output_var is not None:
            y = to_nominal_labels(y)
        d = to_instances(X, y,
                         num_nominal_labels=self._num_nominal_input_labels,
                         num_class_labels=self._num_nominal_output_labels)
        self._classifier.build_classifier(d)
        self.header_ = d.template_instances(d, 0)
        if d.class_attribute.is_nominal:
            self.classes_ = d.class_attribute.values
        else:
            self.classes_ = None
        return self

    def predict(self, X):
        """
        Performs predictions with the trained classifier.

        :param X: the data matrix to generate predictions for, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :return: the score (or scores)
        :rtype: ndarray
        """
        check_is_fitted(self)
        if self._nominal_input_vars is not None:
            X = to_nominal_attributes(X, self._nominal_input_vars)
        X = check_array(X, dtype=None)
        result = []
        for d in X:
            inst = to_instance(self.header_, d, y=missing_value())
            if self.header_.class_attribute.is_nominal:
                result.append(self.header_.class_attribute.value(int(self._classifier.classify_instance(inst))))
            else:
                result.append(self._classifier.classify_instance(inst))
        return np.array(result)

    def predict_proba(self, X):
        """
        Performs predictions and returns class probabilities.

        :param X: the data matrix to generate predictions for, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :return: the probabilities
        """
        check_is_fitted(self)
        if self._nominal_input_vars is not None:
            X = to_nominal_attributes(X, self._nominal_input_vars)
        X = check_array(X, dtype=None)
        result = []
        for d in X:
            inst = to_instance(self.header_, d, y=missing_value())
            result.append(self._classifier.distribution_for_instance(inst))
        return np.array(result)

    def score(self, X, y, sample_weight=None):
        """
        Classification: return the mean accuracy on the given test data and labels.
        Regression: return the coefficient of determination of the prediction.

        :param X: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type X: ndarray
        :param y: the class attribute column, array-like of shape (n_samples,)
        :type y: ndarray
        :param sample_weight: Sample weights. If None, then samples are equally weighted.
        :type sample_weight: array-like of shape (n_samples,), default=None
        :return: the score
        :rtype: float
        """
        y_pred = self.predict(X)
        if self._nominal_output_var:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return r2_score(y, y_pred, sample_weight=sample_weight)

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
        if self._nominal_input_vars is not None:
            result["nominal_input_vars"] = self._nominal_input_vars
        if self._nominal_output_var is not None:
            result["nominal_output_var"] = self._nominal_output_var
        if self._num_nominal_input_labels is not None:
            result["num_nominal_input_labels"] = self._num_nominal_input_labels
        if self._num_nominal_output_labels is not None:
            result["num_nominal_output_labels"] = self._num_nominal_output_labels
        return result

    def set_params(self, **params):
        """
        Sets the options for the classifier, expects 'classname' and 'options'.

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
        self._classifier = Classifier(classname=self._classname, options=self._options)
        self._nominal_input_vars = None
        if "nominal_input_vars" in params:
            self._nominal_input_vars = params["nominal_input_vars"]
        self._nominal_output_var = None
        if "nominal_output_var" in params:
            self._nominal_output_var = params["nominal_output_var"]
        self._num_nominal_input_labels = None
        if "num_nominal_input_labels" in params:
            self._num_nominal_input_labels = params["num_nominal_input_labels"]
        self._num_nominal_output_labels = None
        if "num_nominal_output_labels" in params:
            self._num_nominal_output_labels = params["num_nominal_output_labels"]

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
        result._nominal_input_vars = None if (self._nominal_input_vars is None) else self._nominal_input_vars[:]
        result._nominal_output_var = self._nominal_output_var
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
            return "WekaEstimator(classname='%s', options=%s, nominal_input_vars='%s', nominal_output_var=%s)" % (self._classifier.classname, str(self._classifier.options), str(self._nominal_input_vars), str(self._nominal_output_var))
        else:
            return "WekaEstimator(classname='%s', options=%s, nominal_input_vars=%s, nominal_output_var=%s)" % (self._classifier.classname, str(self._classifier.options), str(self._nominal_input_vars), str(self._nominal_output_var))
