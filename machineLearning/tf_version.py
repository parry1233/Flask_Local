import tensorflow as tf

def tfversion():
    return tf.__version__, tf.config.list_physical_devices('GPU')