/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_OPERATORLIST_H
#define ACL_ARM_COMPUTE_RUNTIME_OPERATORLIST_H

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

/** AddMulAdd
 *
 * Description:
 * Performs a fused Add + Mul + Add [+ Relu-based-Activation] operation.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** ArgMinMaxLayer
 *
 * Description:
 * Function to calculate the index of the minimum or maximum values in a tensor based on an axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ARGMAX
 * ANEURALNETWORKS_ARGMIN
 *
 */

/** ArithmeticAddition
 *
 * Description:
 * Function to add 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ADD
 *
 */

/** ArithmeticSubtraction
 *
 * Description:
 * Function to substract 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SUB
 *
 */

/** BatchNormalizationLayer
 *
 * Description:
 * Function to perform batch normalization.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** BatchToSpaceLayer
 *
 * Description:
 * Batch to space transformation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_BATCH_TO_SPACE_ND
 *
 */

/** BitwiseAnd
 *
 * Description:
 * Function to perform bitwise AND between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_AND
 *
 */

/** BitwiseNot
 *
 * Description:
 * Function to perform bitwise NOT.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_NOT
 *
 */

/** BitwiseOr
 *
 * Description:
 * Function to perform bitwise OR between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOGICAL_OR
 *
 */

/** BitwiseXor
 *
 * Description:
 * Function to perform bitwise XOR between 2 tensors.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** BoundingBoxTransform
 *
 * Description:
 * Transform proposal bounding boxes to target bounding box using bounding box deltas.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** Cast
 *
 * Description:
 * Function to cast a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CAST
 *
 */

/** ChannelShuffleLayer
 *
 * Description:
 * Function to shuffle the channels of the input tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CHANNEL_SHUFFLE
 *
 */

/** Comparison
 *
 * Description:
 * Function to compare 2 tensors.
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
 * Function to transpose the weights for the fully connected layer.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** ConvolutionLayer
 *
 * Description:
 * Function to compute a convolution layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** Conv3D
 *
 * Description:
 * Function to compute a 3d convolution layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_3D
 *
 */

/** Copy
 *
 * Description:
 * Function to copy a tensor.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** Crop
 *
 * Description:
 * Performs a copy of input tensor to the output tensor.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** CropResize
 *
 * Description:
 * Function to perform cropping and resizing.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** DeconvolutionLayer
 *
 * Description:
 * Function to compute a deconvolution or transpose convolution.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** DeconvolutionLayerUpsample
 *
 * Description:
 * Function to execute deconvolution upsample on OpenCL.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** DepthConvertLayer
 *
 * Description:
 * Performs a down-scaling depth conversion.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** DepthToSpaceLayer
 *
 * Description:
 * Depth to Space transformation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEPTH_TO_SPACE
 *
 */

/** DepthwiseConvolutionLayer
 *
 * Description:
 * Function to perform depthwise separable convolution.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEPTHWISE_CONV_2D
 *
 */

/** DequantizationLayer
 *
 * Description:
 * Function to dequantize the values in a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DEQUANTIZE
 *
 */

/** DetectionPostProcessLayer
 *
 * Description:
 * Function to generate the detection output based on center size encoded boxes, class prediction and anchors by doing non maximum suppression (NMS).
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_DETECTION_POSTPROCESSING
 *
 */

/** DirectConvolutionLayer
 *
 * Description:
 * Function to compute direct convolution.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** DirectDeconvolutionLayer
 *
 * Description:
 * Function to run the deconvolution layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** ElementwiseOperations
 *
 * Description:
 * Function to perform in Cpu:
 * - Div
 * - Max
 * - Min
 * - Pow
 * - SquaredDiff
 * - Comparisons (Equal, greater, greater_equal, less, less_equal, not_equal)
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
 * ANEURALNETWORKS_ADD
 * ANEURALNETWORKS_SUB
 * ANEURALNETWORKS_EQUAL
 * ANEURALNETWORKS_GREATER
 * ANEURALNETWORKS_GREATER_EQUAL
 * ANEURALNETWORKS_LESS
 * ANEURALNETWORKS_LESS_EQUAL
 * ANEURALNETWORKS_NOT_EQUAL
 *
 */

/** ElementwiseUnaryLayer
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
 * Fast Fourier Transform 1D.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** FFT2D
 *
 * Description:
 * Fast Fourier Transform 2D.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** FFTConvolutionLayer
 *
 * Description:
 * Fast Fourier Transform Convolution.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** Fill
 *
 * Description:
 * Set the values of a tensor with a given value.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FILL
 *
 */

/** FillBorder
 *
 * Description:
 * Function to fill the borders within the XY-planes.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** FlattenLayer
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
 * Round the value to the lowest number.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FLOOR
 *
 */

/** FullyConnectedLayer
 *
 * Description:
 * Function to perform a fully connected / dense layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_FULLY_CONNECTED
 *
 */

/** FuseBatchNormalization
 *
 * Description:
 * Function to fuse the batch normalization node to a preceding convolution node.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** Gather
 *
 * Description:
 * Performs the Gather operation along the chosen axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_GATHER
 *
 */

/** GEMM
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** GEMMConv2d
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** GEMMConvolutionLayer
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

/** GEMMDeconvolutionLayer
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE_CONV_2D
 *
 */

/** GEMMLowpMatrixMultiplyCore
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** GEMMLowpOutputStage
 *
 * Description:
 * General Matrix Multiplication.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** GenerateProposalsLayer
 *
 * Description:
 * Function to generate proposals for a RPN (Region Proposal Network).
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_GENERATE_PROPOSALS
 *
 */

/** InstanceNormalizationLayer
 *
 * Description:
 * Function to perform a Instance normalization on a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_INSTANCE_NORMALIZATION
 *
 */

/** L2NormalizeLayer
 *
 * Description:
 * Function to perform a L2 normalization on a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_L2_NORMALIZATION
 *
 */

/** Logical
 *
 * Description:
 * Function to perform:
 * - Logical AND
 * - Logical OR
 * - Logical NOT
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** LogicalAnd
 *
 * Description:
 * Function to perform Logical AND.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** LogicalOr
 *
 * Description:
 * Function to perform Logical OR.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** LogicalNot
 *
 * Description:
 * Function to perform Logical NOT.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** LSTMLayer
 *
 * Description:
 * Function to perform a single time step in a Long Short-Term Memory (LSTM) layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LSTM
 *
 */

/** LSTMLayerQuantized
 *
 * Description:
 * Function to perform quantized LSTM (Long Short-Term Memory)
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_QUANTIZED_LSTM
 * ANEURALNETWORKS_QUANTIZED_16BIT_LSTM
 *
 */

/** MatMul
 *
 * Description:
 * Computes a matrix multiplication in batches.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_BATCH_MATMUL
 *
 */

/** MaxUnpoolingLayer
 *
 * Description:
 * Function to perform MaxUnpooling.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** MeanStdDevNormalizationLayer
 *
 * Description:
 * Function to execute mean and standard deviation normalization.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** NormalizationLayer
 *
 * Description:
 * Function to compute normalization layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION
 *
 */

/** NormalizePlanarYUVLayer
 *
 * Description:
 * Function to compute normalization planar YUV layer.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** PadLayer
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
 * Function to perform a multiplication.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_MUL
 *
 */

/** PoolingLayer
 *
 * Description:
 * Function to perform pooling with the specified pooling operation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_AVERAGE_POOL_2D
 * ANEURALNETWORKS_L2_POOL_2D
 * ANEURALNETWORKS_MAX_POOL_2D
 *
 */

/** Pooling3dLayer
 *
 * Description:
 * Function to perform pooling 3D with the specified pooling operation.
 *
 * Equivalent Android NNAPI Op:
 * N/A
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

/** PriorBoxLayer
 *
 * Description:
 * Function to compute prior boxes and clip.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** QLSTMLayer
 *
 * Description:
 * Function to perform quantized LSTM (Long Short-Term Memory).
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_QUANTIZED_LSTM
 * ANEURALNETWORKS_QUANTIZED_16BIT_LSTM
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

/** Range
 *
 * Description:
 * Function to generates a sequence of numbers starting from START and extends by increments of 'STEP' up to but not including 'END'.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** ReduceMean
 *
 * Description:
 * Function to perform reduce mean operation.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_MEAN
 *
 */

/** ReductionOperation
 *
 * Description:
 * Function to perform reduce with the following operations
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

/** ReorderLayer
 *
 * Description:
 * Reorders a tensor to a different weights format.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** ReorgLayer
 *
 * Description:
 * Performs a reorganization layer of input tensor to the output tensor.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** ReshapeLayer
 *
 * Description:
 * Function to reshape a tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RESHAPE
 * ANEURALNETWORKS_SQUEEZE
 *
 */

/** Reverse
 *
 * Description:
 * Function to reverse tensor according to axis.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** RNNLayer
 *
 * Description:
 * Function to perform recurrent neural network layer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RNN
 *
 */

/** ROIAlignLayer
 *
 * Description:
 * Function to perform ROI alignment.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ROI_ALIGN
 *
 */

/** ROIPoolingLayer
 *
 * Description:
 * Function to perform ROI pooling.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_ROI_POOLING
 *
 */

/** Scale
 *
 * Description:
 * Function to perform resize a tensor using to interpolate:
 * - Bilinear
 * - Nearest neighbor
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_RESIZE_BILINEAR
 * ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR
 *
 */

/** Select
 *
 * Description:
 * Function to select values from 2 tensors depending on an input tensor of booleans.
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

/** SoftmaxLayer
 *
 * Description:
 * Function to compute a SoftmaxLayer and a Log SoftmaxLayer.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_LOG_SOFTMAX
 * ANEURALNETWORKS_SOFTMAX
 *
 */

/** SpaceToBatchLayer
 *
 * Description:
 * Function to divide a tensor spatially.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPACE_TO_BATCH_ND
 *
 */

/** SpaceToDepthLayer
 *
 * Description:
 * Function to rearrange blocks of spatial data into depth.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPACE_TO_DEPTH
 *
 */

/** Split
 *
 * Description:
 * Function to split a tensor along a given axis.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_SPLIT
 *
 */

/** StackLayer
 *
 * Description:
 * Function to stack tensors along an axis.
 *
 * Equivalent Android NNAPI Op:
 * n/a
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

/** Tile
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
 * Function to transpose a 2D tensor.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_TRANSPOSE
 *
 */

/** Unstack
 *
 * Description:
 * Function to unpack a rank-R tensor into rank-(R-1) tensors.
 *
 * Equivalent Android NNAPI Op:
 * n/a
 *
 */

/** WinogradConvolutionLayer
 *
 * Description:
 * Function to do Winograd Convolution.
 *
 * Equivalent Android NNAPI Op:
 * ANEURALNETWORKS_CONV_2D
 *
 */

#endif // ACL_ARM_COMPUTE_RUNTIME_OPERATORLIST_H
