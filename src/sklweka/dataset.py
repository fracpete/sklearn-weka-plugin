import math
from scipy.io.arff import loadarff
from weka.core.dataset import Instances, Instance, Attribute
from datetime import datetime
from weka.core.dataset import missing_value
from weka.core.converters import loader_for_file, Loader
import numpy as np
from numpy import ndarray


def parse_range(r, max_value, ordered=True, safe=True):
    """
    Parses the Weka range string (eg "first-last" or "1,3-5,7,10-last")
    of 1-based indices and returns a list of 0-based integers.
    'first' and 'last' are accepted apart from integer strings,
    '-' is used to define a range (low to high).
    The list can be returned ordered or as is.

    :param r: the range string to parse
    :type r: str
    :param max_value: the maximum value for the 1-based indices
    :type max_value: int
    :param ordered: whether to return the list ordered or as is
    :type ordered: bool
    :param safe: whether to catch exceptions or not
    :type safe: bool
    :return: the list of 0-base indices
    :rtype: list
    """
    result = []

    parts = r.replace(" ", "").replace("first", "1").replace("last", str(max_value)).split(",")
    for part in parts:
        if "-" in part:
            bounds = part.split("-")
            low = None
            high = None
            if len(bounds) == 2:
                if safe:
                    try:
                        low = int(bounds[0]) - 1
                        high = int(bounds[1]) - 1
                        if low > high:
                            print("Invalid format for index range ('LOW-HIGH'): %s" % part)
                            continue
                    except:
                        print("Failed to parse bounds of '%s' from range '%s' (max: %d), skipping!" % (part, r, max_value))
                else:
                    low = int(bounds[0]) - 1
                    high = int(bounds[1]) - 1
                    if low > high:
                        print("Invalid format for index range ('LOW-HIGH'): %s" % part)
                        continue
            if (low is not None) and (high is not None):
                for i in range(low, high + 1):
                    result.append(i)
        else:
            if safe:
                try:
                    result.append(int(part) - 1)
                except:
                    print("Failed to parse '%s' from range '%s', skipping (max: %d)!" % (part, r, max_value))
            else:
                result.append(int(part) - 1)

    if ordered:
        result.sort()

    return result


def to_nominal_attributes(X, indices):
    """
    Turns the specified indices numeric column vector into a string vector.

    :param X: the 2D matrix to convert
    :type X: ndarray
    :param indices: the list of 0-based indices of attributes to convert to nominal or range string with 1-based indices
    :type indices: list or str
    :return: the converted matrix
    :rtype: ndarray
    """
    result = []
    if isinstance(indices, str):
        indices = parse_range(indices, X.shape[1])
    indices_set = set(indices)
    for r in X:
        rn = []
        for i in range(len(r)):
            if i in indices_set:
                rn.append("_" + str(r[i]))
            else:
                rn.append(r[i])
        result.append(rn)

    return np.array(result)


def to_nominal_labels(y):
    """
    Turns the numeric column vector into a string vector.

    :param y: the vector to convert
    :type y: list or ndarray
    :return: the converted vector
    :rtype: ndarray
    """
    result = []
    for item in y:
        if isinstance(item, np.bytes_):
            result.append("_" + item.astype(str))
        else:
            result.append("_" + str(item))
    return result


def split_off_class(data, class_index):
    """
    Splits off the class attribute from the data matrix.
    The class index can either be a 0-based int or a 1-based string
    (first,second,last,last-1 are accepted as well).

    :param data: the 2D matrix to process
    :type data: ndarray
    :param class_index: the position of the class attribute to split off
    :type class_index: int or str
    :return: the input variables (2D matrix) and the output variable (1D)
    """

    if len(data) == 0:
        num_atts = -1
    else:
        num_atts = len(data[0])

    # interpret class index
    if isinstance(class_index, str):
        if class_index == "first":
            index = 0
        elif class_index == "second":
            index = 1
        elif class_index == "last-1":
            if num_atts == -1:
                raise Exception("No data, cannot determine # of attributes for class index: %s" % class_index)
            index = num_atts - 2
        elif class_index == "last":
            if num_atts == -1:
                raise Exception("No data, cannot determine # of attributes for class index: %s" % class_index)
            index = num_atts - 1
        else:
            try:
                index = int(class_index) - 1
            except:
                raise Exception("Unsupported class index: %s" % class_index)
    elif isinstance(class_index, int):
        index = class_index
    else:
        raise Exception("Unsupported type for class index: " + str(type(class_index)))

    if index == 0:
        X = [list(r)[1:] for r in data]
    elif index == (num_atts - 1):
        X = [list(r)[0:index] for r in data]
    else:
        X = []
        for r in data:
            r = list(r)
            rn = []
            rn.extend(r[0:index])
            rn.extend(r[index+1:])
            X.append(rn)
    y = [r[index] for r in data]
    return np.array(X), np.array(y)


def load_arff(fname, class_index=None):
    """
    Loads the specified ARFF file. If a class index is provided, either 0-based int or 1-based string
    (first,second,last,last-1 are accepted as well), then the data is split into input variables and
    class attribute.

    :param fname: the path of the ARFF file to load
    :type fname: str
    :param class_index: the class index, either int or str
    :type class_index: int or str
    :return: tuple (X, meta), or in case of a valid class index a tuple (X,y,meta)
    :rtype: tuple
    """
    data, meta = loadarff(fname)
    if class_index is None:
        X = np.array([list(r) for r in data])
        return np.array(X), meta
    else:
        X, y = split_off_class(data, class_index)
        return X, y, meta


def load_dataset(fname, loader=None, class_index=None, internal=False):
    """
    Loads the dataset using Weka's converters. If no loader instance is provided, the extension of
    the file is used to determine a loader (using default options). The data can either be returned
    using mixed types or just numeric (using Weka's internal representation).

    :param fname: the path of the dataset to load
    :type fname: str
    :param loader: the customized Loader instance to use for loading the dataset, can be None
    :type loader: Loader
    :param class_index: the class index string to use ('first', 'second', 'third', 'last-2', 'last-1', 'last' or 1-based index)
    :type class_index: str
    :param internal: whether to return Weka's internal format or mixed data types
    :type internal: bool
    :return: the dataset tuple: (X) if no class index; (X,y) if class index
    """
    if loader is None:
        loader = loader_for_file(fname)
    weka_ds = loader.load_file(fname, class_index=class_index)
    numpy_ds = weka_ds.to_numpy(internal=internal)
    if class_index is not None:
        return split_off_class(numpy_ds, class_index)
    else:
        return numpy_ds


def determine_attribute_types(X):
    """
    Determines the type of the columns.

    :param X: the 2D data to determine the column types for
    :type X: ndarray
    :return: the list of types (C=categorical, N=numeric)
    :rtype: list
    """
    if len(X) == 0:
        raise Exception("No data to convert!")

    num_rows = len(X)
    num_cols = len(X[0])

    # initialize types
    result = []
    for i in range(num_cols):
        result.append("N")

    for i in range(num_cols):
        for n in range(num_rows):
            r = X[n]
            try:
                float(r[i])
            except:
                result[i] = "C"
                break

    return result


def determine_attribute_type(y):
    """
    Determines the type of the column.

    :param y: the 1D vector to determine the type for
    :type y: ndarray
    :return: the type (C=categorical, N=numeric)
    :rtype: str
    """
    result = "N"
    for i in range(len(y)):
        try:
            float(y[i])
        except:
            result = "C"
            break

    return result


def to_instances(X, y=None, att_names=None, att_types=None, class_name=None, class_type=None, relation_name=None,
                 num_nominal_labels=None, num_class_labels=None):
    """
    Turns the 2D matrix and the optional 1D class vector into an Instances object.

    :param X: the input variables, 2D matrix
    :type X: ndarray
    :param y: the optional class value column, 1D vector
    :type y: ndarray
    :param att_names: the list of attribute names
    :type att_names: list
    :param att_types: the list of attribute types (C=categorical, N=numeric), assumes numeric by default if not provided
    :param class_name: the name of the class attribute
    :type class_name: str
    :param class_type: the type of the class attribute (C=categorical, N=numeric)
    :type class_type: str
    :param relation_name: the name for the dataset
    :type relation_name: str
    :param num_nominal_labels: the dictionary with the number of labels (key is 0-based attribute index)
    :type num_nominal_labels: dict
    :param num_class_labels: the number of labels in the class attribute
    :type num_class_labels: int
    :return: the generated Instances object
    :rtype: Instances
    """

    if len(X) == 0:
        raise Exception("No data to convert!")

    # defaults
    if att_types is None:
        att_types = determine_attribute_types(X)
    if att_names is None:
        att_names = []
        for i in range(len(X[0])):
            att_names.append("att-" + str(i+1))
    if relation_name is None:
        relation_name = "scikit-weka @ " + str(datetime.now())
    if class_name is None:
        if "class" not in att_names:
            class_name = "class"
        else:
            class_name = "class-" + str(len(att_names) + 1)
    if y is not None:
        if class_type is None:
            class_type = determine_attribute_type(y)

    # create header
    atts = []

    for i in range(len(X[0])):
        att_name = att_names[i]
        att_type = att_types[i]

        if att_type == "N":
            atts.append(Attribute.create_numeric(att_name))
        elif att_type == "C":
            if (num_nominal_labels is not None) and (i in num_nominal_labels):
                values = []
                for l in range(num_nominal_labels[i]):
                    values.append("_%d" % l)
            else:
                labels = set()
                for n in range(len(X)):
                    r = X[n]
                    v = str(r[i])
                    labels.add(v)
                values = sorted(labels)
            atts.append(Attribute.create_nominal(att_name, values))
        else:
            raise Exception("Unsupported attribute type for column %d: %s" % ((i+1), att_type))

    if y is not None:
        if class_type == "N":
            atts.append(Attribute.create_numeric(class_name))
        elif class_type == "C":
            if num_class_labels is not None:
                values = []
                for l in range(num_class_labels):
                    values.append("_%d" % l)
            else:
                values = sorted(set([str(x) for x in y]))
            atts.append(Attribute.create_nominal(class_name, values))

    result = Instances.create_instances(relation_name, atts, len(X))
    if y is not None:
        result.class_index = result.num_attributes - 1

    # data
    for n in range(len(X)):
        values = []
        r = X[n]
        for i in range(len(r)):
            if att_types[i] == "C":
                values.append(atts[i].index_of(str(r[i])))
            elif att_types[i] == "N":
                values.append(r[i])
            else:
                raise Exception("Unsupported attribute type for column %d: %s" % ((i+1), att_types[i]))
        if y is not None:
            if class_type == "C":
                values.append(atts[-1].index_of(str(y[n])))
            elif class_type == "N":
                values.append(y[n])
            else:
                raise Exception("Unsupported attribute type for class: %s" % class_type)
        inst = Instance.create_instance(values)
        result.add_instance(inst)

    return result


def to_instance(header, x, y=None, weight=1.0):
    """
    Generates an Instance from the data.

    :param header: the data structure to adhere to
    :type header: Instances
    :param x: the 1D vector with input variables
    :type x: ndarray
    :param y: the optional class value
    :type y: object
    :param weight: the weight for the Instance
    :type weight: float
    :return: the generate Instance
    :rtype: Instance
    """
    values = []

    for i in range(len(x)):
        if header.attribute(i).is_nominal:
            values.append(header.attribute(i).index_of(str(x[i])))
        elif header.attribute(i).is_numeric:
            values.append(x[i])
        else:
            raise Exception("Unsupported attribute type for column %d: %s" % ((i+1), header.attribute(i).type_str()))

    if y is not None and header.has_class():
        # missing value?
        if math.isnan(y):
            values.append(missing_value())
        elif header.class_attribute.is_nominal:
            values.append(header.class_attribute.index_of(str(y)))
        elif header.class_attribute.is_numeric:
            values.append(y)
        else:
            raise Exception("Unsupported attribute type for class attribute: %s" % header.class_attribute.type_str())

    result = Instance.create_instance(values, weight=weight)
    result.dataset = header
    return result


def to_array(data):
    """
    Turns the Instances object into ndarrays for X and y. If no class is present, then y will be None.

    :param data: the data to convert
    :type data: Instances
    :return: the generated arrays for X and y
    :rtype: tuple
    """
    has_class = data.has_class()
    class_index = data.class_index
    X = []
    y = [] if has_class else None
    for i in range(data.num_instances):
        inst = data.get_instance(i)
        row = []
        for n in range(data.num_attributes):
            if n == class_index:
                continue
            if data.attribute(n).is_numeric:
                row.append(inst.get_value(n))
            else:
                row.append(inst.get_string_value(n))
        X.append(row)
        if has_class:
            if data.class_attribute.is_numeric:
                y.append(inst.get_value(class_index))
            else:
                y.append(inst.get_string_value(class_index))
    X = np.array(X)
    if y is not None:
        y = np.array(y)
    return X, y
