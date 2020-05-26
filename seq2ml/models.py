"""Model definitions.

To use the fast cuDNN GRU implementation, the following conditions must be met:
    1. activation == tanh
    2. recurrent_activation == sigmoid
    3. recurrent_dropout == 0
    4. unroll is False
    5. use_bias is True
    6. reset_after is True
    7. Inputs are not masked or are strictly right padded


According to the TensorFlow documentation:
    "[cuDNN cannot be used when] Using masking when the input data is not
    strictly right padded (if the mask corresponds to strictly right padded data,
    CuDNN can still be used. This is the most common case)."
https://www.tensorflow.org/guide/keras/rnn#performance_optimization_and_cudnn_kernels_in_tensorflow_20

"""

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


def get_model(name):
    d = {
        "gru_v1": gru_v1,
        "gru_tiny": gru_tiny,
    }
    try:
        return d[name]
    except KeyError:
        raise ValueError(
            "unknown model name: '{}'. Available models: '{}'".format(
                name, "', '".join(model.keys())
            )
        )


def gru_v1(input_shape=(200, 692), dropout_rate=0.5):
    """
    Return GRU RNN compatible with fast cuDNN GRU implementation.
    Variable-length sequences are allowed, but all sequences must have same size in the
    last dimension.

    Parameters
    ----------
    input_shape: int, the shape of input, excluding batch dimension.
    dropout_rate: float in [0, 1], the rate of dropping connections.

    Returns
    -------
    Tensorflow Keras model object.
    """

    model = tfk.Sequential()
    model.add(tfkl.Input(input_shape))
    model.add(tfkl.Masking(mask_value=0.0))
    model.add(tfkl.GRU(256, return_sequences=True))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.GRU(256))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.Dense(128, activation="relu"))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.Dense(1))
    return model


def gru_tiny(input_shape=(200, 692), dropout_rate=0.5):
    """
    Return GRU RNN compatible with fast cuDNN GRU implementation.
    Variable-length sequences are allowed, but all sequences must have same size in the
    last dimension.

    Parameters
    ----------
    input_shape: int, the shape of input, excluding batch dimension.
    dropout_rate: float in [0, 1], the rate of dropping connections.

    Returns
    -------
    Tensorflow Keras model object.
    """

    model = tfk.Sequential()
    model.add(tfkl.Input(input_shape))
    model.add(tfkl.Masking(mask_value=0.0))
    model.add(tfkl.GRU(128))
    model.add(tfkl.Dropout(dropout_rate))
    model.add(tfkl.Dense(1))
    return model
