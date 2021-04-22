/*
 * Copyright (c) 2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_OPERATOR_LIST_H
#define ARM_COMPUTE_OPERATOR_LIST_H

/** ActivationLayer
 *
 * Description:
 * Function to simulate an activation layer with the specified activation function.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ELU
 * ANEURALNETWORKS_HARD_SWISH
 * ANEURALNETWORKS_LOGISTIC
 * ANEURALNETWORKS_RELU
 * ANEURALNETWORKS_RELU1
 * ANEURALNETWORKS_RELU6
 * ANEURALNETWORKS_TANH
 *
 */

/** ArgMinMaxLayer (not ported)
 *
 * Description:
 * Function to calculate the index of the minimum or maximum values in a tensor based on an axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ARGMAX
 * ANEURALNETWORKS_ARGMIN
 *
 */

/** ArithmeticAddition (no CL)
 *
 * Description:
 * Function to add 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ADD
 *
 */

/** ArithmeticSubtraction (no CL)
 *
 * Description:
 * Function to substract 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SUB
 *
 */

/** BatchNormalizationLayer (not ported)
 *
 * Description:
 * @f[ out_i = \gamma * (\frac{in_i - \mu_{B}}{\sqrt{\sigma^2_{B} + \epsilon}}) + \beta \equiv BN_{\gamma,\beta}(in_i) @f]
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** BatchToSpaceLayer (not ported)
 *
 * Description:
 * Rearranges (permutes) data from batch into blocks of spatial data, followed by cropping. It is the reverse transformation of SpaceToBatch (from TF website)
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_BATCH_TO_SPACE_ND
 *
 */

/** BitwiseAnd (not ported)
 *
 * Description:
 * Function to performe bitwise AND between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_AND
 *
 */

/** BitwiseNot (not ported)
 *
 * Description:
 * Function to performe bitwise NOT.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_NOT
 *
 */

/** BitwiseOr (not ported)
 *
 * Description:
 * Function to performe bitwise OR between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_OR
 *
 */

/** BitwiseXor (not ported)
 *
 * Description:
 * Function to performe bitwise XOR between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** BoundingBoxTransform (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ?
 *
 */

/** Cast (not ported)
 *
 * Description:
 * Function to cast a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CAST
 *
 */

/** ChannelShuffelLayer (not ported)
 *
 * Description:
 * Function to cast a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CHANNEL_SHUFFLE
 *
 */

/** Comparison (not ported) (only CL)
 *
 * Description:
 * Function to cast a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_EQUAL
 * ANEURALNETWORKS_GREATER
 * ANEURALNETWORKS_GREATER_EQUAL
 * ANEURALNETWORKS_LESS
 * ANEURALNETWORKS_LESS_EQUAL
 * ANEURALNETWORKS_NOT_EQUAL
 *
 */

/** ConcatenateLayer
 *
 * Description:
 * Function to concatenate tensors along a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONCATENATION
 *
 */

/** ConvertFullyConnectedWeights
 *
 * Description:
 * Function to tranpose the wieghts for the fully connected layer.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** ConvolutionLayer (not ported)
 *
 * Description:
 * Function to compute a convolution layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** Copy
 *
 * Description:
 * Function to copy a tensor.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** Crop (only CL)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ?
 *
 */

/** CropResize (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ?
 *
 */

/** DeconvolutionLayer (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** DeconvolutionLayerUpsample (only CL) (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** DepthConverterLayer (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** DepthToSpaceLayer (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEPTH_TO_SPACE
 *
 */

/** DepthwiseConvolutionLayer (not ported)
 *
 * Description:
 * Function to perform depthwise separable convolution
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEPTHWISE_CONV_2D
 *
 */

/** DequantizationLayer
 *
 * Description:
 * Function to dequantize the values in a tensor
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEQUANTIZE
 *
 */

/** DetectionPostProcessLayer (not ported) (no CL)
 *
 * Description:
 * Function to generate the detection output based on center size encoded boxes, class prediction and anchors by doing non maximum suppression (NMS)
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DETECTION_POSTPROCESSING
 *
 */

/** DirectConvolutionLayer
 *
 * Description:
 * Function to
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** DirectDeconvolutionLayer (only CL)
 *
 * Description:
 * Function to
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** ElementWiseOperations (skip)
 *
 * Description:
 * Function to perform in Cpu:
 * - Div
 * - Max
 * - Min
 * - Pow
 * - SquaredDiff
 * - Comparisons (Equal, greater, greater_equal, less, less_equal, not_equal)
 *
 * Function to perform in CL:
 * - Add
 * - Sub
 * - Div
 * - Max
 * - Min
 * - Pow
 * - SquaredDiff
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_MAXIMUM
 * ANEURALNETWORKS_MINIMUM
 * ANEURALNETWORKS_POW
 * ANEURALNETWORKS_DIV
 * ANEURALNETWORKS_ADD (only CL)
 * ANEURALNETWORKS_SUB (only CL)
 * ANEURALNETWORKS_EQUAL (no CL)
 * ANEURALNETWORKS_GREATER (no CL)
 * ANEURALNETWORKS_GREATER_EQUAL (no CL)
 * ANEURALNETWORKS_LESS (no CL)
 * ANEURALNETWORKS_LESS_EQUAL (no CL)
 * ANEURALNETWORKS_NOT_EQUAL (no CL)
 *
 */

/** ElementWiseOperationUnary (skip)
 *
 * Description:
 * Function to perform:
 * - Rsqrt
 * - Exp
 * - Neg
 * - Log
 * - Abs
 * - Round
 * - Sin
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ABS
 * ANEURALNETWORKS_EXP
 * ANEURALNETWORKS_LOG
 * ANEURALNETWORKS_NEG
 * ANEURALNETWORKS_RSQRT
 * ANEURALNETWORKS_SIN
 *
 */

/** FFT1D
 *
 * Description:
 * Fast Fourier Transform 1D
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** FFT2D
 *
 * Description:
 * Fast Fourier Transform 2D
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** FFTConvolutionLayer
 *
 * Description:
 * Fast Fourier Transform Convolution
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** Fill
 *
 * Description:
 * Set the values of a tensor with a given value
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FILL
 *
 */

/** FillBorder (not ported)
 *
 * Description:
 *
 *
 * Equivalent Android NNAPI Op:
 * ?
 *
 */

/** FlattenLayer (not ported)
 *
 * Description:
 * Reshape a tensor to be 1D
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RESHAPE
 *
 */

/** Floor
 *
 * Description:
 * Round the value to the lowest number
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FLOOR
 *
 */

/** FullyConnectedLayer (not ported)
 *
 * Description:
 * Function to perform a fully connected / dense layer
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FULLY_CONNECTED
 *
 */

/** FuseBatchNormalization (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** Gather (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_GATHER
 *
 */

/** GEMM (not ported)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GEMMConv2D (not ported) (no CL)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GEMMConvolutionLayer (not ported)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GEMMDeconvolutionLayer (not ported) (only CL)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GEMMLowpMatrixMultiplyCore (not ported)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GEMMLowpOutputStage (not ported)
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** GenerateProposalsLayer (not ported)
 *
 * Description:
 * Function to generate proposals for a RPN (Region Proposal Network).
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_GENERATE_PROPOSALS
 *
 */

/** InstanceNormalizationLayer (not ported)
 *
 * Description:
 * Function to perform a Instance normalization on a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_INSTANCE_NORMALIZATION
 *
 */

/** L2NormalizationLayer (not ported)
 *
 * Description:
 * Function to perform a L2 normalization on a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_L2_NORMALIZATION
 *
 */

/** Logical (no CL)
 *
 * Description:
 * Function to perform:
 * - Logical AND
 * - Logical OR
 * - Logical NOT
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** LogicalAnd (only CL)
 *
 * Description:
 * Function to perform Logical AND
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** LogicalOr (only CL)
 *
 * Description:
 * Function to perform Logical OR
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** LogicalNot (only CL)
 *
 * Description:
 * Function to perform Logical NOT
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** LSTMLayer (not ported)
 *
 * Description:
 * Function to perform LSTM
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LSTM
 *
 */

/** LSTMLayerQuantized (not ported)
 *
 * Description:
 * Function to perform LSTM
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_QUANTIZED_LSTM
 * ANEURALNETWORKS_QUANTIZED_16BIT_LSTM ?
 *
 */

/** MaxUnpoolingLayer (not ported)
 *
 * Description:
 * Function to perform MaxUnpooling
 *
 * Equivalent Android NNAPI Op:
 *  ?
 *
 */

/** MeanStdDevNormalizationLayer (not ported)
 *
 * Description:
 * Function to execute mean and standard deviation normalization.
 *
 * Equivalent Android NNAPI Op:
 * None ?
 *
 */

/** MeanStdDevNormalizationLayer (not ported)
 *
 * Description:
 * Function to execute mean and standard deviation normalization.
 *
 * Equivalent Android NNAPI Op:
 * None ?
 *
 */

/** NormalizationLayer (not ported)
 *
 * Description:
 * Function to compute normalization layer.
 *
 * Equivalent Android NNAPI Op:
 * None ?
 *
 */

/** PadLayer (not ported)
 *
 * Description:
 * Function to pad a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_PAD
 * ANEURALNETWORKS_PAD_V2
 *
 */

/** Permute
 *
 * Description:
 * Function to transpose an ND tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE
 *
 */

/** PixelWiseMultiplication
 *
 * Description:
 * Function to performe a multiplication.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_MUL
 *
 */

/** PoolingLayer
 *
 * Description:
 * Function to performe pooling with the specified pooling operation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_AVERAGE_POOL_2D
 * ANEURALNETWORKS_L2_POOL_2D
 * ANEURALNETWORKS_MAX_POOL_2D
 *
 */

/** PReluLayer
 *
 * Description:
 * Function to compute the activation layer with the PRELU activation function.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_PRELU
 *
 */

/** PriorBoxLayer (not ported)
 *
 * Description:
 * Function to compute the activation layer with the PRELU activation function.
 *
 * Equivalent Android NNAPI Op:
 * ?
 *
 */

/** QLSTMLayer (not ported)
 *
 * Description:
 * Function to perform LSTM
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_QUANTIZED_LSTM
 * ANEURALNETWORKS_QUANTIZED_16BIT_LSTM ?
 *
 */

/** QuantizationLayer
 *
 * Description:
 * Function to perform quantization layer
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_QUANTIZE
 *
 */

/** Range (not ported)
 *
 * Description:
 * Function to .
 *
 * Equivalent Android NNAPI Op:
 * none?
 *
 */

/** RecudeMean (not ported)
 *
 * Description:
 * Function to performe reduce mean operation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_MEAN
 *
 */

/** RecudeOperation (not ported)
 *
 * Description:
 * Function to performe reduce mean operation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_REDUCE_ALL
 * ANEURALNETWORKS_REDUCE_ANY
 * ANEURALNETWORKS_REDUCE_MAX
 * ANEURALNETWORKS_REDUCE_MIN
 * ANEURALNETWORKS_REDUCE_PROD
 * ANEURALNETWORKS_REDUCE_SUM
 *
 */

/** RecudeOperation (not ported)
 *
 * Description:
 * Function to performe reduce with the following operations
 * - ARG_IDX_MAX: Index of the max value
 * - ARG_IDX_MIN: Index of the min value
 * - MEAN_SUM:    Mean of sum
 * - PROD:        Product
 * - SUM_SQUARE:  Sum of squares
 * - SUM:         Sum
 * - MIN:         Min
 * - MAX:         Max
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_REDUCE_ALL
 * ANEURALNETWORKS_REDUCE_ANY
 * ANEURALNETWORKS_REDUCE_MAX
 * ANEURALNETWORKS_REDUCE_MIN
 * ANEURALNETWORKS_REDUCE_PROD
 * ANEURALNETWORKS_REDUCE_SUM
 *
 */

/** ReorgLayer (not ported)
 *
 * Description:
 * Function to performe reorg
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** ReshapeLayer
 *
 * Description:
 * Fucntion to reshape a tensor
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RESHAPE
 * ANEURALNETWORKS_SQUEEZE
 *
 */

/** ReverseLayer (not ported)
 *
 * Description:
 * Fucntion to .
 *
 * Equivalent Android NNAPI Op:
 * None?
 *
 */

/** RNNLayer (not ported)
 *
 * Description:
 * Fucntion to perform RNN .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RNN
 *
 */

/** ROIAligmentLayer (not ported)
 *
 * Description:
 * Fucntion to perform RNN .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ROI_ALIGN
 *
 */

/** ROIPoolingLayer (not ported)
 *
 * Description:
 * Fucntion to perform RNN .
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ROI_POOLING
 *
 */

/** Scale
 *
 * Description:
 * Fucntion to perform resize a tensor using to interpolate:
 * - Bilenear
 * - Nearest neighbor
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RESIZE_BILINEAR
 * ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR
 *
 */

/** Select (not ported)
 *
 * Description:
 * Fucntion to select values from 2 tensors depending on an input tensor of booleans.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SELECT
 *
 */

/** Slice
 *
 * Description:
 * Function to perform tensor slicing.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SLICE
 *
 */

/** SoftmaxLayer (skip)
 *
 * Description:
 * Function to compute a SoftmaxLayer and a Log SoftmaxLayer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOG_SOFTMAX
 * ANEURALNETWORKS_SOFTMAX
 *
 */

/** SpaceToBatchLayer (not ported)
 *
 * Description:
 * Function to divide a tensor spatially.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPACE_TO_BATCH_ND
 *
 */

/** SpaceToDepthLayer (not ported)
 *
 * Description:
 * Function to rearrange blocks of spatial data into depth.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPACE_TO_DEPTH
 *
 */

/** Split (not ported)
 *
 * Description:
 * Function to split a tensor along a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPLIT
 *
 */

/** StackLayer (not ported)
 *
 * Description:
 * Function to stack tensors along an axis.
 *
 * Equivalent Android NNAPI Op:
 * none
 *
 */

/** StridedSlice
 *
 * Description:
 * Function to extract a strided slice of a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_STRIDED_SLICE
 *
 */

/** Tile  (not ported)
 *
 * Description:
 * Function to construct a tensor by tiling a given tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TILE
 *
 */

/** Transpose
 *
 * Description:
 * Function to transpose an 2D tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE
 *
 */

/** Unstack (not ported)
 *
 * Description:
 * Function to unpack a rank-R tensor into rank-(R-1) tensors.
 *
 * Equivalent Android NNAPI Op:
 * none
 *
 */

/** WinogradConvolutionLayer (not ported)
 *
 * Description:
 * Function to.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

/** WinogradInputTransform (not ported) (only CL)
 *
 * Description:
 * Function to.
 *
 * Equivalent Android NNAPI Op:
 * None
 *
 */

#endif /* ARM_COMPUTE_OPERATOR_LIST_H */