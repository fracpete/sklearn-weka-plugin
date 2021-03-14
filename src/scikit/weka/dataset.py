from scipy.io.arff import loadarff
from weka.core.dataset import Instances, Instance, Attribute
from datetime import datetime


def split_off_class(data, class_index):
    """
    Splits off the class attribute from the data matrix.
    The class index can either be a 0-based int or a 1-based string
    (first,second,last,last-1 are accepted as well).

    :param data: the 2D matrix to process
    :type data: ndarrau
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
    return X, y


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
        return data, meta
    X, y = split_off_class(data, class_index)
    return X, y, meta


def to_instances(X, y=None, att_names=None, att_types=None, class_name=None, class_type=None, relation_name=None):
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
    :return: the generated Instances object
    :rtype: Instances
    """

    if len(X) == 0:
        raise Exception("No data to convert!")

    # defaults
    if att_types is None:
        att_types = []
        for i in range(len(X[0])):
            att_types.append("N")
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
    if class_type is None:
        class_type = "N"

    # create header
    atts = []

    for i in range(len(X[0])):
        att_name = att_names[i]
        att_type = att_types[i]

        if att_type == "N":
            atts.append(Attribute.create_numeric(att_name))
        elif att_type == "C":
            values = []
            for n in range(len(X)):
                r = X[n]
                values.append(str(r[i]))
            atts.append(Attribute.create_nominal(att_name, values))
        else:
            raise Exception("Unsupported attribute type for column %d: %s" % ((i+1), att_type))

    if y is not None:
        if class_type == "N":
            atts.append(Attribute.create_numeric(class_name))
        elif class_type == "C":
            values = [str(x) for x in y]
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
                values.append(atts[i].index_of(str(y[n])))
            elif class_type == "N":
                values.append(y[n])
            else:
                raise Exception("Unsupported attribute type for class: %s" % att_types[i])
        inst = Instance.create_instance(values)
        result.add_instance(inst)

    return result


def to_instance(header, x, y=None, weight=1.0):
    """

    :param header: the data structure to adhere to
    :type header: Instances
    :param x: the 1D vector with input variables
    :type x: ndarray
    :param y: the optional class value
    :type y: object
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
        if header.class_attribute.is_nominal:
            values.append(header.class_attribute.index_of(str(y)))
        elif header.class_attribute.is_numeric:
            values.append(y)
        else:
            raise Exception("Unsupported attribute type for class attribute: %s" % header.class_attribute.type_str())

    return Instance.create_instance(values, weight=weight)
