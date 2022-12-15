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
  arm_compute_validation
  PRIVATE UNIT/SafeIntegerOps.cpp
          UNIT/Version.cpp
          UNIT/TensorInfo.cpp
          UNIT/TensorShape.cpp
          UNIT/Utils.cpp
          UNIT/SubTensorInfo.cpp
          UNIT/WindowIterator.cpp
          UNIT/LifetimeManager.cpp
          UNIT/GPUTarget.cpp
          CPP/DetectionPostProcessLayer.cpp
          CPP/TopKV.cpp
          CPP/DFT.cpp
          CPP/Permute.cpp
          CPP/NonMaximumSuppression.cpp)

if(ENABLE_NEON)
  target_sources(
    arm_compute_validation
    PRIVATE NEON/ElementwiseNegation.cpp
            NEON/BoundingBoxTransform.cpp
            NEON/ChannelShuffle.cpp
            NEON/Logical.cpp
            NEON/DilatedConvolutionLayer.cpp
            NEON/PoolingLayer.cpp
            NEON/BitwiseNot.cpp
            NEON/FillBorder.cpp
            NEON/ElementwiseRsqrtLayer.cpp
            NEON/DepthConcatenateLayer.cpp
            NEON/ElementwisePower.cpp
            NEON/Fill.cpp
            NEON/ROIPoolingLayer.cpp
            NEON/LSTMLayer.cpp
            NEON/ArithmeticSubtraction.cpp
            NEON/GEMMLowp.cpp
            NEON/Unstack.cpp
            NEON/Slice.cpp
            NEON/Pooling3dLayer.cpp
            NEON/BitwiseOr.cpp
            NEON/HeightConcatenateLayer.cpp
            NEON/ReshapeLayer.cpp
            NEON/SoftmaxLayer.cpp
            NEON/Gather.cpp
            NEON/CropResize.cpp
            NEON/ReductionOperation.cpp
            NEON/PixelWiseMultiplication.cpp
            NEON/LogSoftmaxLayer.cpp
            NEON/DepthConvertLayer.cpp
            NEON/Flatten.cpp
            NEON/ElementwiseKernelSelection.cpp
            NEON/DepthToSpaceLayer.cpp
            NEON/ElementwiseAbsoluteValue.cpp
            NEON/PadLayer.cpp
            NEON/MeanStdDevNormalizationLayer.cpp
            NEON/GlobalPoolingLayer.cpp
            NEON/RNNLayer.cpp
            NEON/DetectionPostProcessLayer.cpp
            NEON/ElementwiseRound.cpp
            NEON/BitwiseXor.cpp
            NEON/GEMM.cpp
            NEON/FuseBatchNormalization.cpp
            NEON/BitwiseAnd.cpp
            NEON/ElementwiseMax.cpp
            NEON/ReduceMean.cpp
            NEON/Reverse.cpp
            NEON/L2NormalizeLayer.cpp
            NEON/Convolution3D.cpp
            NEON/ArithmeticAddition.cpp
            NEON/ActivationLayer.cpp
            NEON/SpaceToBatchLayer.cpp
            NEON/ElementwiseLog.cpp
            NEON/LSTMLayerQuantized.cpp
            NEON/Im2Col.cpp
            NEON/DequantizationLayer.cpp
            NEON/DeconvolutionLayer.cpp
            NEON/Select.cpp
            NEON/ElementwiseSin.cpp
            NEON/PReluLayer.cpp
            NEON/BatchNormalizationLayer.cpp
            NEON/ElementwiseMin.cpp
            NEON/InstanceNormalizationLayer.cpp
            NEON/ROIAlignLayer.cpp
            NEON/ElementwiseDivision.cpp
            NEON/ElementwiseExpLayer.cpp
            NEON/ArgMinMax.cpp
            NEON/QLSTMLayerNormalization.cpp
            NEON/Col2Im.cpp
            NEON/Split.cpp
            NEON/Transpose.cpp
            NEON/GenerateProposalsLayer.cpp
            NEON/StackLayer.cpp
            NEON/WidthConcatenateLayer.cpp
            NEON/NormalizationLayer.cpp
            NEON/Copy.cpp
            NEON/ElementwiseSquareDiff.cpp
            NEON/MaxUnpoolingLayer.cpp
            NEON/Permute.cpp
            NEON/Comparisons.cpp
            NEON/BatchConcatenateLayer.cpp
            NEON/Tile.cpp
            NEON/BatchToSpaceLayer.cpp
            NEON/SpaceToDepthLayer.cpp
            NEON/DepthwiseConvolutionLayerNative.cpp
            NEON/QuantizationLayer.cpp
            NEON/ConvertFullyConnectedWeights.cpp
            NEON/Floor.cpp
            NEON/FFT.cpp
            NEON/Cast.cpp
            NEON/PriorBoxLayer.cpp
            NEON/Scale.cpp
            NEON/ReorgLayer.cpp
            NEON/Range.cpp
            NEON/DirectConvolutionLayer.cpp
            NEON/DepthwiseConvolutionLayer.cpp
            NEON/FullyConnectedLayer.cpp
            NEON/ConvolutionLayer.cpp
            NEON/StridedSlice.cpp
            NEON/UNIT/DynamicTensor.cpp
            NEON/UNIT/TensorAllocator.cpp
            NEON/UNIT/MemoryManager.cpp
            NEON/UNIT/RuntimeContext.cpp)
endif()
