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
#ifndef __ARM_COMPUTE_NEKERNELS_H__
#define __ARM_COMPUTE_NEKERNELS_H__

/* Header regrouping all the NEON kernels */
#include "arm_compute/core/NEON/kernels/NEAbsoluteDifferenceKernel.h"
#include "arm_compute/core/NEON/kernels/NEAccumulateKernel.h"
#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticSubtractionKernel.h"
#include "arm_compute/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEBatchToSpaceLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEBitwiseAndKernel.h"
#include "arm_compute/core/NEON/kernels/NEBitwiseNotKernel.h"
#include "arm_compute/core/NEON/kernels/NEBitwiseOrKernel.h"
#include "arm_compute/core/NEON/kernels/NEBitwiseXorKernel.h"
#include "arm_compute/core/NEON/kernels/NEBox3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NECannyEdgeKernel.h"
#include "arm_compute/core/NEON/kernels/NEChannelCombineKernel.h"
#include "arm_compute/core/NEON/kernels/NEChannelExtractKernel.h"
#include "arm_compute/core/NEON/kernels/NEChannelShuffleLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NECol2ImKernel.h"
#include "arm_compute/core/NEON/kernels/NEColorConvertKernel.h"
#include "arm_compute/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "arm_compute/core/NEON/kernels/NEConvolutionKernel.h"
#include "arm_compute/core/NEON/kernels/NECopyKernel.h"
#include "arm_compute/core/NEON/kernels/NECropKernel.h"
#include "arm_compute/core/NEON/kernels/NECumulativeDistributionKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthConvertLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseVectorToTensorKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseWeightsReshapeKernel.h"
#include "arm_compute/core/NEON/kernels/NEDequantizationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDerivativeKernel.h"
#include "arm_compute/core/NEON/kernels/NEDilateKernel.h"
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"
#include "arm_compute/core/NEON/kernels/NEElementwiseOperationKernel.h"
#include "arm_compute/core/NEON/kernels/NEElementwiseUnaryKernel.h"
#include "arm_compute/core/NEON/kernels/NEErodeKernel.h"
#include "arm_compute/core/NEON/kernels/NEFFTDigitReverseKernel.h"
#include "arm_compute/core/NEON/kernels/NEFFTRadixStageKernel.h"
#include "arm_compute/core/NEON/kernels/NEFFTScaleKernel.h"
#include "arm_compute/core/NEON/kernels/NEFastCornersKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillArrayKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillInnerBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEFlattenLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEFloorKernel.h"
#include "arm_compute/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixVectorMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEGatherKernel.h"
#include "arm_compute/core/NEON/kernels/NEGaussian3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGaussian5x5Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGaussianPyramidKernel.h"
#include "arm_compute/core/NEON/kernels/NEHOGDescriptorKernel.h"
#include "arm_compute/core/NEON/kernels/NEHOGDetectorKernel.h"
#include "arm_compute/core/NEON/kernels/NEHarrisCornersKernel.h"
#include "arm_compute/core/NEON/kernels/NEHeightConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEHistogramKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NEIntegralImageKernel.h"
#include "arm_compute/core/NEON/kernels/NEL2NormalizeLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NELKTrackerKernel.h"
#include "arm_compute/core/NEON/kernels/NELocallyConnectedMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEMagnitudePhaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEMeanStdDevKernel.h"
#include "arm_compute/core/NEON/kernels/NEMedian3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NEMemsetKernel.h"
#include "arm_compute/core/NEON/kernels/NEMinMaxLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEMinMaxLocationKernel.h"
#include "arm_compute/core/NEON/kernels/NENonLinearFilterKernel.h"
#include "arm_compute/core/NEON/kernels/NENonMaximaSuppression3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NENormalizationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEPermuteKernel.h"
#include "arm_compute/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/NEON/kernels/NEPoolingLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEPriorBoxLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEQuantizationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEROIPoolingLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NERangeKernel.h"
#include "arm_compute/core/NEON/kernels/NEReductionOperationKernel.h"
#include "arm_compute/core/NEON/kernels/NERemapKernel.h"
#include "arm_compute/core/NEON/kernels/NEReorgLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEReverseKernel.h"
#include "arm_compute/core/NEON/kernels/NEScaleKernel.h"
#include "arm_compute/core/NEON/kernels/NEScharr3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NESelectKernel.h"
#include "arm_compute/core/NEON/kernels/NESobel3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NESobel5x5Kernel.h"
#include "arm_compute/core/NEON/kernels/NESobel7x7Kernel.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NESpaceToBatchLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEStackLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEStridedSliceKernel.h"
#include "arm_compute/core/NEON/kernels/NETableLookupKernel.h"
#include "arm_compute/core/NEON/kernels/NEThresholdKernel.h"
#include "arm_compute/core/NEON/kernels/NETileKernel.h"
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"
#include "arm_compute/core/NEON/kernels/NEUpsampleLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEWarpKernel.h"
#include "arm_compute/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "arm_compute/core/NEON/kernels/NEWidthConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEWinogradConvolutionLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEYOLOLayerKernel.h"

#endif /* __ARM_COMPUTE_NEKERNELS_H__ */
