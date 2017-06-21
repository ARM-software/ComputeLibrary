#!/usr/bin/env python
import argparse

import caffe
import numpy as np
import scipy.io


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser('Extract CNN hyper-parameters')
    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Caffe model file')
    parser.add_argument('-n', dest='netFile', type=str, required=True, help='Caffe netlist')
    args = parser.parse_args()

    # Create Caffe Net
    net = caffe.Net(args.netFile, 1, weights=args.modelFile)

    # Read and dump blobs
    for name, blobs in net.params.iteritems():
        print 'Name: {0}, Blobs: {1}'.format(name, len(blobs))
        for i in range(len(blobs)):
            # Weights
            if i == 0:
                outname = name + "_w"
            # Bias
            elif i == 1:
                outname = name + "_b"
            else:
                pass

            print("%s : %s" % (outname, blobs[i].data.shape))
            # Dump as binary
            blobs[i].data.tofile(outname + ".dat")
