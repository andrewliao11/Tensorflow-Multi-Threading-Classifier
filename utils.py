import tensorflow as tf

def get_session_config(fraction=0.5, num_threads=1):
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction=fraction
    conf.intra_op_parallelism_threads=num_threads
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    return conf
