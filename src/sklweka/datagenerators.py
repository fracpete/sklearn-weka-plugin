import numpy as np
from weka.datagenerators import DataGenerator


def generate_data(generator, att_names=False):
    """
    Generates data using the Weka data generator.

    :param generator: the data generator to use
    :type generator: DataGenerator
    :param att_names: whether to return the attribute names as well
    :type att_names: bool
    :return: tuple of X and y matrices or tuple of X, y, X_names, y_name
    :rtype: tuple
    """

    format = generator.define_data_format()
    data = generator.generate_examples()
    index = data.class_index
    X = []
    X_names = []
    y = [] if (index > -1) else None
    y_name = None
    att_types = []
    for i in range(data.num_attributes):
        if i == index:
            y_name = data.attribute(i).name
        else:
            X_names.append(data.attribute(i).name)
        if data.attribute(i).is_numeric:
            att_types.append("N")
        else:
            att_types.append("S")
    for inst in data:
        row = []
        for i in range(data.num_attributes):
            if att_types[i] == "N":
                value = inst.get_value(i)
            else:
                value = inst.get_string_value(i)
            if i == index:
                y.append(value)
            else:
                row.append(value)
        X.append(row)

    if att_names:
        if y is None:
            return np.asarray(X), None, X_names, None
        else:
            return np.asarray(X), np.asarray(y), X_names, y_name
    else:
        if y is None:
            return np.asarray(X), None
        else:
            return np.asarray(X), np.asarray(y)
