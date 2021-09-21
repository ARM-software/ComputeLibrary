#!/usr/bin/env python3
# Copyright (c) 2021 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
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
import json
import logging
import os
import sys
from argparse import ArgumentParser

import tflite

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from utils.model_identification import identify_model_type
from utils.tflite_helpers import tflite_op2acl, tflite_typecode2name, tflite_typecode2aclname

SUPPORTED_MODEL_TYPES = ["tflite"]
logger = logging.getLogger("report_model_ops")


def get_ops_types_from_tflite_graph(model):
    """
    Helper function that extract operator related meta-data from a TFLite model

    Parameters
        ----------
    model: str
        Respective TFLite model to analyse

    Returns
    ----------
    supported_ops, unsupported_ops, data_types: tuple
        A tuple with the sets of unique operator types and data-types that are present in the model
    """

    logger.debug(f"Analysing TFLite mode '{model}'!")

    with open(model, "rb") as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    # Extract unique operators
    nr_unique_ops = model.OperatorCodesLength()
    unique_ops = {tflite.opcode2name(model.OperatorCodes(op_id).BuiltinCode()) for op_id in range(0, nr_unique_ops)}

    # Extract IO data-types
    supported_data_types = set()
    unsupported_data_types = set()
    for subgraph_id in range(0, model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_id)
        for tensor_id in range(0, subgraph.TensorsLength()):
            try:
                supported_data_types.add(tflite_typecode2aclname(subgraph.Tensors(tensor_id).Type()))
            except ValueError:
                unsupported_data_types.add(tflite_typecode2name(subgraph.Tensors(tensor_id).Type()))
                logger.warning(f"Data type {tflite_typecode2name(subgraph.Tensors(tensor_id).Type())} is not supported by ComputeLibrary")

    # Perform mapping between TfLite ops to ComputeLibrary ones
    supported_ops = set()
    unsupported_ops = set()
    for top in unique_ops:
        try:
            supported_ops.add(tflite_op2acl(top))
        except ValueError:
            unsupported_ops.add(top)
            logger.warning(f"Operator {top} does not have ComputeLibrary mapping")

    return (supported_ops, unsupported_ops, supported_data_types, unsupported_data_types)


def extract_model_meta(model, model_type):
    """
    Function that calls the appropriate model parser to extract model related meta-data
    Supported parsers: TFLite

    Parameters
        ----------
    model: str
        Path to model that we want to analyze
    model_type:
        type of the model

    Returns
    ----------
    ops, data_types: (tuple)
        A tuple with the list of unique operator types and data-types that are present in the model
    """

    if model_type == "tflite":
        return get_ops_types_from_tflite_graph(model)
    else:
        logger.warning(f"Model type '{model_type}' is unsupported!")
        return ()


def generate_build_config(ops, data_types, data_layouts):
    """
    Function that generates a compatible ComputeLibrary operator-based build configuration

    Parameters
        ----------
    ops: set
        Set with the operators to add in the build configuration
    data_types:
        Set with the data types to add in the build configuration
    data_layouts:
        Set with the data layouts to add in the build configuration

    Returns
    ----------
    config_data: dict
        Dictionary compatible with ComputeLibrary
    """
    config_data = {}
    config_data["operators"] = list(ops)
    config_data["data_types"] = list(data_types)
    config_data["data_layouts"] = list(data_layouts)

    return config_data


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Report map of operations in a list of models.
            The script consumes deep learning models and reports the type of operations and data-types used
            Supported model types: TFLite """
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        required=True,
        type=str,
        help=f"List of models; supported model types: {SUPPORTED_MODEL_TYPES}",
    )
    parser.add_argument("-D", "--debug", action="store_true", help="Enable script debugging output")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="JSON configuration file used that can be used for custom ComputeLibrary builds",
    )
    args = parser.parse_args()

    # Setup Logger
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)

    # Extract operator mapping
    final_supported_ops = set()
    final_unsupported_ops = set()
    final_supported_dts = set()
    final_unsupported_dts = set()
    final_layouts = {"nhwc"} # Data layout for TFLite is always NHWC
    for model in args.models:
        logger.debug(f"Starting analyzing {model} model")

        model_type = identify_model_type(model)
        supported_model_ops, unsupported_mode_ops, supported_model_dts, unsupported_model_dts = extract_model_meta(model, model_type)
        final_supported_ops.update(supported_model_ops)
        final_unsupported_ops.update(unsupported_mode_ops)
        final_supported_dts.update(supported_model_dts)
        final_unsupported_dts.update(unsupported_model_dts)

    logger.info("=== Supported Operators")
    logger.info(final_supported_ops)
    if(len(final_unsupported_ops)):
        logger.info("=== Unsupported Operators")
        logger.info(final_unsupported_ops)
    logger.info("=== Data Types")
    logger.info(final_supported_dts)
    if(len(final_unsupported_dts)):
        logger.info("=== Unsupported Data Types")
        logger.info(final_unsupported_dts)
    logger.info("=== Data Layouts")
    logger.info(final_layouts)

    # Generate JSON file
    if args.config:
        logger.debug("Generating JSON build configuration file")
        config_data = generate_build_config(final_supported_ops, final_supported_dts, final_layouts)
        with open(args.config, "w") as f:
            json.dump(config_data, f)
