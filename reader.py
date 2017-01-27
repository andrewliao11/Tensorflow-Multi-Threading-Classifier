import tensorflow as tf

def read_to_tensor(bin_filepath, num_epochs=None):

    # construct a FIFOQueue containing a list of filenames
    filename_queue = tf.train.string_input_producer([bin_filepath], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([28*28], tf.int64)
        })
    # now return the converted data
    label = features['label']
    image = features['image']
    return label, image
