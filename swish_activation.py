import tensorflow as tf

"""
reference : https://arxiv.org/pdf/1710.05941.pdf
"""
def swish(x):
    """
    :param : x is a Tensor with type `float32`, `float64`, `int32`
    :return : x*sigmoid(x)
    """
    return x*tf.sigmoid(x)