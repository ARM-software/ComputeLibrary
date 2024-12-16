#!/usr/bin/env python
#
# SPDX-FileCopyrightText: 2018, 2024 Arm Limited
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Extract trainable parameters from a frozen model and stores them in numpy arrays.
Usage:
    python tf_frozen_model_extractor -m path_to_frozem_model -d path_to_store_the_parameters

Saves each variable to a {variable_name}.npy binary file.

Note that the script permutes the trainable parameters to NCHW format. This is a pretty manual step thus it's not thoroughly tested.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

strings_to_remove=["read", "/:0"]
permutations = { 1 : [0], 2 : [1, 0], 3 : [2, 1, 0], 4 : [3, 2, 0, 1]}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract TensorFlow net parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to TensorFlow frozen graph file (.pb)')
    parser.add_argument('-d', dest='dumpPath', type=str, required=False, default='./', help='Path to store the resulting files.')
    parser.add_argument('--nostore', dest='storeRes', action='store_false', help='Specify if files should not be stored. Used for debugging.')
    parser.set_defaults(storeRes=True)
    args = parser.parse_args()

    # Create directory if not present
    if not os.path.exists(args.dumpPath):
        os.makedirs(args.dumpPath)

    # Extract parameters
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            print("Loading model.")
            with gfile.FastGFile(args.modelFile, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="", op_dict=None, producer_op_list=None)

                for op in graph.get_operations():
                    for op_val in op.values():
                        varname = op_val.name

                        # Skip non-const values
                        if "read" in varname:
                            t  = op_val.eval()
                            tT = t.transpose(permutations[len(t.shape)])
                            t  = np.ascontiguousarray(tT)

                            for s in strings_to_remove:
                                varname = varname.replace(s, "")
                            if os.path.sep in varname:
                                varname = varname.replace(os.path.sep, '_')
                                print("Renaming variable {0} to {1}".format(op_val.name, varname))

                            # Store files
                            if args.storeRes:
                                print("Saving variable {0} with shape {1} ...".format(varname, t.shape))
                                np.save(os.path.join(args.dumpPath, varname), t)
