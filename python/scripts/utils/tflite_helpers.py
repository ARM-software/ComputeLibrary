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

_TFLITE_TYPECODE2NAME = {
    0: "Float32",
    1: "Float16",
    2: "Int32",
    3: "Uint8",
    4: "Int64",
    5: "String",
    6: "Bool",
    7: "Int16",
    8: "Complex64",
    9: "Int8",
}

_TFLITE_TO_ACL = {
    "ADD": "Add",  # 0
    "AVERAGE_POOL_2D": "Pool2d",  # 1
    "CONCATENATION": "Concatenate",  # 2
    "CONV_2D": "Conv2d",  # 3
    "DEPTHWISE_CONV_2D": "DepthwiseConv2d",  # 4
    "DEPTH_TO_SPACE": "DepthToSpace",  # 5
    "DEQUANTIZE": "Dequantize",  # 6
    # "EMBEDDING_LOOKUP" : "Unsupported",                  #7
    "FLOOR": "Floor",  # 8
    "FULLY_CONNECTED": "FullyConnected",  # 9
    # "HASHTABLE_LOOKUP" : "Unsupported",                  #10
    "L2_NORMALIZATION": "L2Normalize",  # 11
    "L2_POOL_2D": "Pool2d",  # 12
    "LOCAL_RESPONSE_NORMALIZATION": "Normalize",  # 13
    "LOGISTIC": "Activation",  # 14
    # "LSH_PROJECTION" : "Unsupported",                    #15
    "LSTM": "LSTM",  # 16
    "MAX_POOL_2D": "Pool2d",  # 17
    "MUL": "Mul",  # 18
    "RELU": "Activation",  # 19
    "RELU_N1_TO_1": "Activation",  # 20
    "RELU6": "Activation",  # 21
    "RESHAPE": "Reshape",  # 22
    "RESIZE_BILINEAR": "Scale",  # 23
    "RNN": "RNN",  # 24
    "SOFTMAX": "Softmax",  # 25
    "SPACE_TO_DEPTH": "SpaceToDepth",  # 26
    # "SVDF" : "Unsupported",                              #27
    "TANH": "Activation",  # 28
    # "CONCAT_EMBEDDINGS" : "Unsupported",                 #29
    # "SKIP_GRAM" : "Unsupported",                         #30
    # "CALL" : "Unsupported",                              #31
    # "CUSTOM" : "Unsupported",                            #32
    # "EMBEDDING_LOOKUP_SPARSE" : "Unsupported",           #33
    "PAD": "Pad",  # 34
    # "UNIDIRECTIONAL_SEQUENCE_RNN" : "Unsupported",       #35
    "GATHER": "Gather",  # 36
    "BATCH_TO_SPACE_ND": "BatchToSpace",  # 37
    "SPACE_TO_BATCH_ND": "SpaceToBatch",  # 38
    "TRANSPOSE": "Permute",  # 39
    "MEAN": "Reduction",  # 40
    "SUB": "Sub",  # 41
    "DIV": "Div",  # 42
    "SQUEEZE": "Reshape",  # 43
    # "UNIDIRECTIONAL_SEQUENCE_LSTM" : "Unsupported",      #44
    "STRIDED_SLICE": "StridedSlice",  # 45
    # "BIDIRECTIONAL_SEQUENCE_RNN" : "Unsupported",        #46
    "EXP": "ElementwiseUnary",  # 47
    # "TOPK_V2" : "Unsupported",                           #48
    "SPLIT": "Split",  # 49
    "LOG_SOFTMAX": "Softmax",  # 50
    # "DELEGATE" : "Unuspported",                          #51
    # "BIDIRECTIONAL_SEQUENCE_LSTM" : "Unsupported",       #52
    "CAST": "Cast",  # 53
    "PRELU": "PRelu",  # 54
    "MAXIMUM": "ElementwiseBinary",  # 55
    "ARG_MAX": "Reduction",  # 56
    "MINIMUM": "ElementwiseBinary",  # 57
    "LESS": "ElementwiseBinary",  # 58
    "NEG": "ElementwiseUnary",  # 59
    "PADV2": "Pad",  # 60
    "GREATER": "ElementwiseBinary",  # 61
    "GREATER_EQUAL": "ElementwiseBinary",  # 62
    "LESS_EQUAL": "ElementwiseBinary",  # 63
    "SELECT": "Select",  # 64
    "SLICE": "Slice",  # 65
    "SIN": "ElementwiseUnary",  # 66
    "TRANSPOSE_CONV": "TransposeConv2d",  # 67
    # "SPARSE_TO_DENSE" : "Unsupported",                   #68
    "TILE": "Tile",  # 69
    "EXPAND_DIMS": "Reshape",  # 70
    "EQUAL": "ElementwiseBinary",  # 71
    "NOT_EQUAL": "ElementwiseBinary",  # 72
    "LOG": "ElementwiseUnary",  # 73
    "SUM": "Reduction",  # 74
    "SQRT": "Activation",  # 75
    "RSQRT": "ElementwiseUnary",  # 76
    "SHAPE": "",  # 77
    "POW": "ElementwiseBinary",  # 78
    "ARG_MIN": "Reduction",  # 79
    # "FAKE_QUANT" : "Unsupported",                        #80
    "REDUCE_PROD": "Reduction",  # 81
    "REDUCE_MAX": "Reduction",  # 82
    "PACK": "Stack",  # 83
    "LOGICAL_OR": "ElementwiseBinary",  # 84
    "ONE_HOT": "Unsupported",  # 85
    "LOGICAL_AND": "ElementwiseBinary",  # 86
    "LOGICAL_NOT": "ElementwiseUnary",  # 87
    "UNPACK": "Unstack",  # 88
    "REDUCE_MIN": "Reduction",  # 89
    # "FLOOR_DIV" :  "Unsupported",                        #90
    # "REDUCE_ANY" :  "Unsupported",                       #91
    "SQUARE": "Activation",  # 92
    "ZEROS_LIKE": "",  # 93
    "FILL": "Fill",  # 94
    # "FLOOR_MOD" :  "Unsupported",                        #95
    "RANGE": "",  # 96
    "RESIZE_NEAREST_NEIGHBOR": "Scale",  # 97
    "LEAKY_RELU": "Activation",  # 98
    "SQUARED_DIFFERENCE": "ElementwiseBinary",  # 99
    "MIRROR_PAD": "Pad",  # 100
    "ABS": "ElementwiseUnary",  # 101
    "SPLIT_V": "Split",  # 102
    # "UNIQUE" :  "Unsupported",                           #103
    # "CEIL" :  "Unsupported",                             #104
    "REVERSE_V2": "Reverse",  # 105
    "ADD_N": "Add",  # 106
    "GATHER_ND": "Gather",  # 107
    # "COS" :  "Unsupported",                              #108
    # "WHERE" :  "Unsupported",                            #109
    "RANK": "",  # 110
    "ELU": "Activation",  # 111
    # "REVERSE_SEQUENCE" : "Unsupported",                  #112
    # "MATRIX_DIAG" : "Unsupported",                       #113
    "QUANTIZE": "Quantize",  # 114
    # "MATRIX_SET_DIAG" :  "Unsupported",                  #115
    "ROUND": "ElementwiseUnary",  # 116
    "HARD_SWISH": "Activation",  # 117
    # "IF" :  "Unsupported",                               #118
    # "WHILE" :  "Unsupported",                            #119
    # "NON_MAX_SUPPRESSION_V4" :  "Unsupported",           #120
    # "NON_MAX_SUPPRESSION_V5" :  "Unsupported",           #121
    # "SCATTER_ND" :  "Unsupported",                       #122
    "SELECT_V2": "Select",  # 123
    "DENSIFY": "Cast",  # 124
    # "SEGMENT_SUM" : "Unsupported",                       #125
    "BATCH_MATMUL": "GEMM",  # 126
    # "PLACEHOLDER_FOR_GREATER_OP_CODES" :  "Unsupported", #127
    # "CUMSUM" :  "Unsupported",                           #128
    # "CALL_ONCE" : "Unsupported",                         #129
    # "BROADCAST_TO" : "Unsupported",                      #130
    # "RFFT2D" :  "Unsupported",                           #131
    # "CONV_3D" :  "Unsupported",                          #132
    # "IMAG" : "Unsupported",                              #133
    # "REAL" : "Unsupported",                              #134
    # "COMPLEX_ABS" : "Unsupported",                       #135
    # "HASHTABLE" :  "Unsupported",                        #136
    # "HASHTABLE_FIND" :  "Unsupported",                   #137
    # "HASHTABLE_IMPORT" :  "Unsupported",                 #138
    # "HASHTABLE_SIZE" :  "Unsupported",                   #139
    # "REDUCE_ALL" :  "Unsupported",                       #140
    # "CONV_3D_TRANSPOSE" : "Unsupported",                 #141
    # "VAR_HANDLE" :  "Unsupported",                       #142
    # "READ_VARIABLE" :  "Unsupported",                    #143
    # "ASSIGN_VARIABLE" :  "Unsupported",                  #144
}


def tflite_typecode2name(toc):
    """Stringify TfLite data-type opcodes

    Parameters:
    ----------
    toc: int
        TfLite type opcode

    Returns
    ----------
    str
        Stringified opcode

    Raises
    ------
    ValueError
        If opcode does not exist in the map
    """
    if toc in _TFLITE_TYPECODE2NAME:
        return _TFLITE_TYPECODE2NAME[toc]
    else:
        raise ValueError("Unknown typecode %d" % toc)


def tflite_op2acl(top):
    """Map TfLite operators to ComputeLibrary ones

    Parameters:
    ----------
    top: str
        TfLite operator name

    Returns
    ----------
    str
        Relevant ComputeLibrary operator name

    Raises
    ------
    ValueError
        If operator cannot be mapped
    """
    if top in _TFLITE_TO_ACL:
        return _TFLITE_TO_ACL[top]
    else:
        raise ValueError("Operator {} does not exist in ComputeLibrary" % top)
