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

"""Extracts mnist image data from the Caffe data files and stores them in numpy arrays
Usage
    python caffe_mnist_image_extractor.py -d path_to_caffe_data_directory -o desired_output_path

Saves the first 10 images extracted as input10.npy, the first 100 images as input100.npy, and the
corresponding labels to labels100.txt.

Tested with Caffe 1.0 on Python 2.7
"""
import argparse
import os
import struct
import numpy as np
from array import array


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract Caffe mnist image data')
    parser.add_argument('-d', dest='dataDir', type=str, required=True, help='Path to Caffe data directory')
    parser.add_argument('-o', dest='outDir', type=str, default='.', help='Output directory (default = current directory)')
    args = parser.parse_args()

    images_filename = os.path.join(args.dataDir, 'mnist/t10k-images-idx3-ubyte')
    labels_filename = os.path.join(args.dataDir, 'mnist/t10k-labels-idx1-ubyte')

    images_file = open(images_filename, 'rb')
    labels_file = open(labels_filename, 'rb')
    images_magic, images_size, rows, cols = struct.unpack('>IIII', images_file.read(16))
    labels_magic, labels_size = struct.unpack('>II', labels_file.read(8))
    images = array('B', images_file.read())
    labels = array('b', labels_file.read())

    input10_path   = os.path.join(args.outDir, 'input10.npy')
    input100_path  = os.path.join(args.outDir, 'input100.npy')
    labels100_path = os.path.join(args.outDir, 'labels100.npy')

    outputs_10  = np.zeros(( 10, 28, 28, 1), dtype=np.float32)
    outputs_100 = np.zeros((100, 28, 28, 1), dtype=np.float32)
    labels_output = open(labels100_path, 'w')
    for i in xrange(100):
        image = np.array(images[i * rows * cols : (i + 1) * rows * cols]).reshape((rows, cols)) / 256.0
        outputs_100[i, :, :, 0] = image

        if i < 10:
            outputs_10[i, :, :, 0] = image

        if i == 10:
            np.save(input10_path, np.transpose(outputs_10, (0, 3, 1, 2)))
            print "Wrote", input10_path

        labels_output.write(str(labels[i]) + '\n')

    labels_output.close()
    print "Wrote", labels100_path

    np.save(input100_path, np.transpose(outputs_100, (0, 3, 1, 2)))
    print "Wrote", input100_path
