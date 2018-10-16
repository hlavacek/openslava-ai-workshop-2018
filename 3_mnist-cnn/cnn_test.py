import numpy as np


def test_cnn_input_shape(scale_function):
    zeros = np.zeros((1000, 28, 28))
    assert np.array_equal(scale_function(zeros).shape[1:], [28, 28, 1])
    print('Test OK')
