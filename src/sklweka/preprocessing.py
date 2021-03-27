import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from weka.filters import Filter
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.dataset import missing_value, Instances
from weka.core.serialization import deepcopy
from sklweka.dataset import to_instances, to_array, to_nominal_attributes, to_nominal_labels


class WekaTransformer(BaseEstimator, OptionHandler, TransformerMixin):
    """
    Wraps a Weka filter within the scikit-learn framework.
    """

    def __init__(self, jobject=None, filter=None, classname=None, options=None):
        """
        Initializes the estimator. Can be either instantiated via the following priority of parameters:
        1. JB_Object representing a Java Filter object
        2. Filter pww3 wrapper
        3. classname/options

        :param jobject: the JB_Object representing a Weka filter to use
        :type jobject: JB_Object
        :param filter: the filter wrapper to use
        :type filter: Filter
        :param classname: the classname of the Weka filter to instantiate
        :type classname: str
        :param options: the command-line options of the Weka filter to instantiate
        :type options: list
        """
        if jobject is not None:
            _jobject = jobject
        elif filter is not None:
            _jobject = filter.jobject
        elif classname is not None:
            if options is None:
                options = []
            classifier = Filter(classname=classname, options=options)
            _jobject = classifier.jobject
        else:
            raise Exception("At least Java classname must be provided!")

        if not is_instance_of(_jobject, "weka.filters.Filter"):
            raise Exception("Java object does not implement weka.filters.Filter!")

        super(WekaTransformer, self).__init__(_jobject)
        self._filter = Filter(jobject=_jobject)
        self.header_ = None
        # the following references are required for get_params/set_params
        self._classname = classname
        self._options = options

    @property
    def filter(self):
        """
        Returns the underlying filter object, if any.

        :return: the classifier object
        :rtype: Classifier
        """
        return self._filter

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
        :param targets: the optional class attribute column, array-like of shape (n_samples,)
        :type targets: ndarray
        :return: itself
        :rtype: WekaTransformer
        """
        if targets is None:
            check_array(data)
        else:
            check_X_y(data, targets)
        d = to_instances(data, y=targets)
        self.header_ = Instances.template_instances(d)
        self._filter.inputformat(d)
        self._filter.filter(d)
        return self

    def transform(self, data, targets=None):
        """
        Filters the data.

        :param data: the data to filter, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: the optional class attribute column, array-like of shape (n_samples,)
        :type targets: ndarray
        :return: the filtered data, X if no targets or (X, y) if targets provided
        :rtype: ndarray or tuple
        """
        check_is_fitted(self)
        no_targets = targets is None

        # dummy class values necessary?
        if no_targets and self.header_.has_class():
            targets = []
            for i in range(data.shape[0]):
                targets.append(missing_value())
            targets = np.array(targets)

        d = to_instances(data, y=targets)
        d_new = self._filter.filter(d)
        X, y = to_array(d_new)
        if no_targets:
            return X
        else:
            return X, y

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
        self._filter = Filter(classname=self._classname, options=self._options)

    def __str__(self):
        """
        For printing the model.

        :return: the model representation, if any
        :rtype: str
        """
        if self._filter is None:
            return self._classname + ": No filter instantiated yet"
        else:
            return str(self._filter)

    def __copy__(self):
        """
        Creates a deep copy of itself.

        :return: the copy
        :rtype: WekaTransformer
        """
        result = WekaTransformer(jobject=deepcopy(self.jobject))
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
        return "WekaTransformer(classname='%s', options=%s)" % (self._filter.classname, str(self._filter.options))


class MakeNominal(BaseEstimator, TransformerMixin):
    """
    Converts numeric columns to nominal ones (ie string labels).
    """

    def __init__(self, input_vars=None, output_var=False):
        """
        Initializes the estimator.

        :param input_vars: the list of 0-based indices to convert
        :type input_vars: list
        :param output_var: whether to convert the output variable as well
        :type output_var: bool
        """
        super(MakeNominal, self).__init__()
        self._input_vars = None if input_vars is None else input_vars[:]
        self._output_var = output_var

    @property
    def input_vars(self):
        """
        Returns the 0-based indices of the input variables to convert.

        :return: the indices, can be None
        :rtype: list
        """
        return self._input_vars

    @property
    def output_vars(self):
        """
        Returns whether the output variable gets converted as well.

        :return: True if the output variable gets converted
        :rtype: bool
        """
        return self._input_vars

    def fit(self, data, targets):
        """
        Trains the estimator.

        :param data: the input variables as matrix, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: the optional class attribute column, array-like of shape (n_samples,)
        :type targets: ndarray
        :return: itself
        :rtype: WekaTransformer
        """
        if targets is None:
            check_array(data)
        else:
            check_X_y(data, targets)
        self.initialized_ = True
        return self

    def transform(self, data, targets=None):
        """
        Filters the data.

        :param data: the data to filter, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :param targets: the optional class attribute column, array-like of shape (n_samples,)
        :type targets: ndarray
        :return: the filtered data, X if no targets or (X, y) if targets provided
        :rtype: ndarray or tuple
        """
        check_is_fitted(self)
        X_new = to_nominal_attributes(data, self._input_vars)
        y_new = None
        if targets is not None:
            y_new = to_nominal_labels(targets)
        return X_new, y_new

    def get_params(self, deep=True):
        """
        Returns the parameters for this classifier, basically classname and options list.

        :param deep: ignored
        :type deep: bool
        :return: the dictionary with options
        :rtype: dict
        """
        result = dict()
        result["input_vars"] = self._input_vars
        result["output_var"] = self._output_var
        return result

    def set_params(self, **params):
        """
        Sets the options for the classifier, expects 'classname' and 'options'.

        :param params: the parameter dictionary
        :type params: dict
        """
        if "input_vars" not in params:
            raise Exception("Cannot find 'input_vars' in parameters!")
        if "output_var" not in params:
            raise Exception("Cannot find 'output_var' in parameters!")
        self._input_vars = params["input_vars"]
        self._output_var = params["output_vars"]

    def __str__(self):
        """
        For printing the model.

        :return: the model representation, if any
        :rtype: str
        """
        if self._filter is None:
            return self._classname + ": No filter instantiated yet"
        else:
            return str(self._filter)

    def __copy__(self):
        """
        Creates a deep copy of itself.

        :return: the copy
        :rtype: WekaTransformer
        """
        return MakeNominal(
            input_vars=(None if self._input_vars is None else self._input_vars[:]),
            output_var=self._output_var)

    def __repr__(self, N_CHAR_MAX=700):
        """
        Returns a valid Python string using its classname and options.

        :param N_CHAR_MAX: ignored
        :type N_CHAR_MAX: int
        :return: the representation
        :rtype: str
        """
        return "MakeNominal(input_vars=%s, output_var=%s)" % (repr(self._input_vars), str(self._output_var))
