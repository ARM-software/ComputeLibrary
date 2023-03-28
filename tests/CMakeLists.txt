# Copyright (c) 2023 Arm Limited.
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

target_sources(
  arm_compute_validation_framework
  PRIVATE validation/Validation.cpp
          validation/Helpers.cpp
          validation/reference/BoundingBoxTransform.cpp
          validation/reference/GEMMReshapeRHSMatrix.cpp
          validation/reference/ChannelShuffle.cpp
          validation/reference/Logical.cpp
          validation/reference/PoolingLayer.cpp
          validation/reference/BitwiseNot.cpp
          validation/reference/Conv3D.cpp
          validation/reference/GEMMReshapeLHSMatrix.cpp
          validation/reference/ComputeAllAnchors.cpp
          validation/reference/DepthConcatenateLayer.cpp
          validation/reference/TableLookup.cpp
          validation/reference/ROIPoolingLayer.cpp
          validation/reference/SliceOperations.cpp
          validation/reference/GEMMLowp.cpp
          validation/reference/Unstack.cpp
          validation/reference/Pooling3dLayer.cpp
          validation/reference/BitwiseOr.cpp
          validation/reference/ReshapeLayer.cpp
          validation/reference/SoftmaxLayer.cpp
          validation/reference/Gather.cpp
          validation/reference/Utils.cpp
          validation/reference/Accumulate.cpp
          validation/reference/CropResize.cpp
          validation/reference/ReductionOperation.cpp
          validation/reference/ConcatenateLayer.cpp
          validation/reference/PixelWiseMultiplication.cpp
          validation/reference/DepthConvertLayer.cpp
          validation/reference/Erode.cpp
          validation/reference/DepthToSpaceLayer.cpp
          validation/reference/PadLayer.cpp
          validation/reference/MeanStdDevNormalizationLayer.cpp
          validation/reference/BitwiseXor.cpp
          validation/reference/GEMM.cpp
          validation/reference/NormalizePlanarYUVLayer.cpp
          validation/reference/FuseBatchNormalization.cpp
          validation/reference/BitwiseAnd.cpp
          validation/reference/SpaceToDepth.cpp
          validation/reference/NonMaximaSuppression.cpp
          validation/reference/Reverse.cpp
          validation/reference/DFT.cpp
          validation/reference/L2NormalizeLayer.cpp
          validation/reference/ActivationLayer.cpp
          validation/reference/SpaceToBatch.cpp
          validation/reference/PostOps.cpp
          validation/reference/Im2Col.cpp
          validation/reference/DequantizationLayer.cpp
          validation/reference/DeconvolutionLayer.cpp
          validation/reference/MinMaxLocation.cpp
          validation/reference/Select.cpp
          validation/reference/BatchNormalizationLayer.cpp
          validation/reference/InstanceNormalizationLayer.cpp
          validation/reference/ROIAlignLayer.cpp
          validation/reference/ElementwiseUnary.cpp
          validation/reference/MeanStdDev.cpp
          validation/reference/QLSTMLayerNormalization.cpp
          validation/reference/Col2Im.cpp
          validation/reference/FlattenLayer.cpp
          validation/reference/AbsoluteDifference.cpp
          validation/reference/Transpose.cpp
          validation/reference/StackLayer.cpp
          validation/reference/NormalizationLayer.cpp
          validation/reference/Copy.cpp
          validation/reference/MaxUnpoolingLayer.cpp
          validation/reference/Winograd.cpp
          validation/reference/Permute.cpp
          validation/reference/Comparisons.cpp
          validation/reference/Tile.cpp
          validation/reference/BatchToSpaceLayer.cpp
          validation/reference/ElementwiseOperations.cpp
          validation/reference/QuantizationLayer.cpp
          validation/reference/NonMaxSuppression.cpp
          validation/reference/WeightsReshape.cpp
          validation/reference/ArithmeticOperations.cpp
          validation/reference/ConvertFullyConnectedWeights.cpp
          validation/reference/Floor.cpp
          validation/reference/PriorBoxLayer.cpp
          validation/reference/Scale.cpp
          validation/reference/ReorgLayer.cpp
          validation/reference/Range.cpp
          validation/reference/ArithmeticDivision.cpp
          validation/reference/DepthwiseConvolutionLayer.cpp
          validation/reference/FullyConnectedLayer.cpp
          validation/reference/ConvolutionLayer.cpp
          validation/reference/Reorder.cpp
          framework/Framework.cpp
          framework/Utils.cpp
          framework/Exceptions.cpp
          framework/DatasetModes.cpp
          framework/TestFilter.cpp
          framework/Profiler.cpp
          framework/ParametersLibrary.cpp
          framework/command_line/CommonOptions.cpp
          framework/instruments/WallClockTimer.cpp
          framework/instruments/InstrumentsStats.cpp
          framework/instruments/Instruments.cpp
          framework/instruments/SchedulerTimer.cpp
          framework/instruments/hwc_names.hpp
          framework/instruments/hwc.hpp
          framework/printers/PrettyPrinter.cpp
          framework/printers/Printer.cpp
          framework/printers/JSONPrinter.cpp
          framework/printers/Printers.cpp
          AssetsLibrary.cpp
          RawTensor.cpp
          main.cpp)
