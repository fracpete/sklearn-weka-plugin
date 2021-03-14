from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from weka.classifiers import Classifier
from weka.core.classes import is_instance_of, OptionHandler
from weka.core.dataset import missing_value
from weka.core.serialization import deepcopy
from scikit.weka.dataset import to_instances, to_instance


class WekaEstimator(BaseEstimator, OptionHandler, RegressorMixin, ClassifierMixin):

    def __init__(self, jobject=None, classifier=None, classname=None, options=None):
        """
        Initializes the estimator. Can be either instantiated via the following priority of parameters:
        1. JB_Object representing a Java Classifier object
        2. Classifier pww3 wrapper
        3. classname/options

        :param classname: the classname of the Weka classifier to instantiate
        :type classname: str
        :param options: the command-line options of the Weka classifier to instantiate
        :type options: str
        :param classifier: the classifier wrapper to use
        :type classifier: Classifier
        :param jobject: the JB_Object representing a Weka classifier to use
        :type jobject: JB_Object
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
        self._classname = classname
        self._options = options
        self._header = None

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
        return self._header

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
        d = to_instances(data, targets)
        self._classifier.build_classifier(d)
        self._header = d.template_instances(d, 0)
        return self

    def predict(self, data):
        """
        Performs predictions with the trained classifier.

        :param data: the data matrix to generate predictions for, array-like of shape (n_samples, n_features)
        :type data: ndarray
        :return:
        """

        # scoring with list of rows?
        if isinstance(data, list):
            if len(data) > 0:
                if isinstance(data[0], list):
                    result = []
                    for d in data:
                        result.append(self.predict(d))
                    return result

        inst = to_instance(self._header, data, missing_value())
        if self._header.class_attribute.is_nominal:
            return self._classifier.distribution_for_instance(inst)
        else:
            return self._classifier.classify_instance(inst)

    def get_params(self, deep=True):
        """
        Returns the parameters for this classifier, basically classname and options list.

        :param deep: ignored
        :type deep: bool
        :return: the dictionary with options
        :rtype: dict
        """
        result = {}
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
            raise Exception("Cannot find 'optionms' in parameters!")
        self._classifier = Classifier(classname=params["classname"], options=params["options"])

    def __str__(self):
        """
        For printing the model.

        :return: the model representation, if any
        :rtype: str
        """
        if self._classifier is None:
            return "No model built yet"
        else:
            return str(self._classifier)

    def __copy__(self):
        """
        Creates a deep copy of itself.

        :return: the copy
        :rtype: WekaEstimator
        """
        return WekaEstimator(jobject=deepcopy(self.jobject))

    def __repr__(self, N_CHAR_MAX=700):
        """
        Returns a valid Python string using its classname and options.

        :param N_CHAR_MAX: ignored
        :type N_CHAR_MAX: int
        :return: the representation
        :rtype: str
        """
        return "WekaEstimator(classname='%s', options=%s)" % (self._classifier.classname, str(self._classifier.options))
