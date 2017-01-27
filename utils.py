import tensorflow as tf

def get_session_config(fraction):
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = fraction
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    return conf
