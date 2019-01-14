/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFUNCTIONS_H__
#define __ARM_COMPUTE_NEFUNCTIONS_H__

/* Header regrouping all the NEON functions */
#include "arm_compute/runtime/NEON/functions/NEAbsoluteDifference.h"
#include "arm_compute/runtime/NEON/functions/NEAccumulate.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArgMinMaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseAnd.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseNot.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseOr.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseXor.h"
#include "arm_compute/runtime/NEON/functions/NEBox3x3.h"
#include "arm_compute/runtime/NEON/functions/NECannyEdge.h"
#include "arm_compute/runtime/NEON/functions/NEChannelCombine.h"
#include "arm_compute/runtime/NEON/functions/NEChannelExtract.h"
#include "arm_compute/runtime/NEON/functions/NEChannelShuffleLayer.h"
#include "arm_compute/runtime/NEON/functions/NECol2Im.h"
#include "arm_compute/runtime/NEON/functions/NEColorConvert.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/NEON/functions/NEConvolution.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NECopy.h"
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseSeparableConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDerivative.h"
#include "arm_compute/runtime/NEON/functions/NEDilate.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"
#include "arm_compute/runtime/NEON/functions/NEEqualizeHistogram.h"
#include "arm_compute/runtime/NEON/functions/NEErode.h"
#include "arm_compute/runtime/NEON/functions/NEFastCorners.h"
#include "arm_compute/runtime/NEON/functions/NEFillBorder.h"
#include "arm_compute/runtime/NEON/functions/NEFlattenLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFloor.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMInterleave4x4.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpAssemblyMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMTranspose1xW.h"
#include "arm_compute/runtime/NEON/functions/NEGather.h"
#include "arm_compute/runtime/NEON/functions/NEGaussian3x3.h"
#include "arm_compute/runtime/NEON/functions/NEGaussian5x5.h"
#include "arm_compute/runtime/NEON/functions/NEGaussianPyramid.h"
#include "arm_compute/runtime/NEON/functions/NEHOGDescriptor.h"
#include "arm_compute/runtime/NEON/functions/NEHOGDetector.h"
#include "arm_compute/runtime/NEON/functions/NEHOGGradient.h"
#include "arm_compute/runtime/NEON/functions/NEHOGMultiDetection.h"
#include "arm_compute/runtime/NEON/functions/NEHarrisCorners.h"
#include "arm_compute/runtime/NEON/functions/NEHistogram.h"
#include "arm_compute/runtime/NEON/functions/NEIm2Col.h"
#include "arm_compute/runtime/NEON/functions/NEIntegralImage.h"
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"
#include "arm_compute/runtime/NEON/functions/NELSTMLayer.h"
#include "arm_compute/runtime/NEON/functions/NELaplacianPyramid.h"
#include "arm_compute/runtime/NEON/functions/NELaplacianReconstruct.h"
#include "arm_compute/runtime/NEON/functions/NELocallyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEMagnitude.h"
#include "arm_compute/runtime/NEON/functions/NEMeanStdDev.h"
#include "arm_compute/runtime/NEON/functions/NEMedian3x3.h"
#include "arm_compute/runtime/NEON/functions/NEMinMaxLocation.h"
#include "arm_compute/runtime/NEON/functions/NENonLinearFilter.h"
#include "arm_compute/runtime/NEON/functions/NENonMaximaSuppression3x3.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEOpticalFlow.h"
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEPhase.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPriorBoxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NERNNLayer.h"
#include "arm_compute/runtime/NEON/functions/NEROIPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NERange.h"
#include "arm_compute/runtime/NEON/functions/NEReduceMean.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NERemap.h"
#include "arm_compute/runtime/NEON/functions/NEReorgLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReverse.h"
#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/NEON/functions/NEScharr3x3.h"
#include "arm_compute/runtime/NEON/functions/NESelect.h"
#include "arm_compute/runtime/NEON/functions/NESimpleAssemblyFunction.h"
#include "arm_compute/runtime/NEON/functions/NESlice.h"
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/NEON/functions/NESobel5x5.h"
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NESplit.h"
#include "arm_compute/runtime/NEON/functions/NEStackLayer.h"
#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"
#include "arm_compute/runtime/NEON/functions/NETableLookup.h"
#include "arm_compute/runtime/NEON/functions/NEThreshold.h"
#include "arm_compute/runtime/NEON/functions/NETile.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/NEON/functions/NEUnstack.h"
#include "arm_compute/runtime/NEON/functions/NEUpsampleLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWarpAffine.h"
#include "arm_compute/runtime/NEON/functions/NEWarpPerspective.h"
#include "arm_compute/runtime/NEON/functions/NEWidthConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEYOLOLayer.h"

#endif /* __ARM_COMPUTE_NEFUNCTIONS_H__ */
