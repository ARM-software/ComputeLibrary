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
import logging
import os


def is_tflite_model(model_path):
    """Check if a model is of TfLite type

    Parameters:
    ----------
    model_path: str
        Path to model

    Returns
    ----------
    bool:
        True if given path is a valid TfLite model
    """

    try:
        with open(model_path, "rb") as f:
            hdr_bytes = f.read(8)
            hdr_str = hdr_bytes[4:].decode("utf-8")
            if hdr_str == "TFL3":
                return True
            else:
                return False
    except:
        return False


def identify_model_type(model_path):
    """Identify the type of a given deep learning model

    Parameters:
    ----------
    model_path: str
        Path to model

    Returns
    ----------
    model_type: str
        String representation of model type or 'None' if type could not be retrieved.
    """

    if not os.path.exists(model_path):
        logging.warn(f"Provided model {model_path} does not exist!")
        return None

    if is_tflite_model(model_path):
        model_type = "tflite"
    else:
        logging.warn(logging.warn(f"Provided model {model_path} is not of supported type!"))
        model_type = None

    return model_type
