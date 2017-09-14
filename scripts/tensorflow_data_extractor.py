#!/usr/bin/env python
"""Extracts trainable parameters from Tensorflow models and stores them in numpy arrays.
Usage
    python tensorflow_data_extractor -m path_to_binary_checkpoint_file -n path_to_metagraph_file

Saves each variable to a {variable_name}.npy binary file.

Note that since Tensorflow version 0.11 the binary checkpoint file which contains the values for each parameter has the format of:
    {model_name}.data-{step}-of-{max_step}
instead of:
    {model_name}.ckpt
When dealing with binary files with version >= 0.11, only pass {model_name} to -m option;
when dealing with binary files with version < 0.11, pass the whole file name {model_name}.ckpt to -m option.

Also note that this script relies on the parameters to be extracted being in the
'trainable_variables' tensor collection. By default all variables are automatically added to this collection unless
specified otherwise by the user. Thus should a user alter this default behavior and/or want to extract parameters from other
collections, tf.GraphKeys.TRAINABLE_VARIABLES should be replaced accordingly.

Tested with Tensorflow 1.2, 1.3 on Python 2.7.6 and Python 3.4.3.
"""
import argparse
import numpy as np
import os
import tensorflow as tf


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract Tensorflow net parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to Tensorflow checkpoint binary\
            file. For Tensorflow version >= 0.11, only include model name; for Tensorflow version < 0.11, include\
            model name with ".ckpt" extension')
    parser.add_argument('-n', dest='netFile', type=str, required=True, help='Path to Tensorflow MetaGraph file')
    args = parser.parse_args()

    # Load Tensorflow Net
    saver = tf.train.import_meta_graph(args.netFile)
    with tf.Session() as sess:
        # Restore session
        saver.restore(sess, args.modelFile)
        print('Model restored.')
        # Save trainable variables to numpy arrays
        for t in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            varname = t.name
            if os.path.sep in t.name:
                varname = varname.replace(os.path.sep, '_')
                print("Renaming variable {0} to {1}".format(t.name, varname))
            print("Saving variable {0} with shape {1} ...".format(varname, t.shape))
            # Dump as binary
            np.save(varname, sess.run(t))
