/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEKERNELS_H
#define ARM_COMPUTE_NEKERNELS_H

/* Header regrouping all the Neon kernels */
#include "src/core/NEON/kernels/NEAbsoluteDifferenceKernel.h"
#include "src/core/NEON/kernels/NEAccumulateKernel.h"
#include "src/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"
#include "src/core/NEON/kernels/NEBatchToSpaceLayerKernel.h"
#include "src/core/NEON/kernels/NEBitwiseAndKernel.h"
#include "src/core/NEON/kernels/NEBitwiseNotKernel.h"
#include "src/core/NEON/kernels/NEBitwiseOrKernel.h"
#include "src/core/NEON/kernels/NEBitwiseXorKernel.h"
#include "src/core/NEON/kernels/NEBoundingBoxTransformKernel.h"
#include "src/core/NEON/kernels/NEBox3x3Kernel.h"
#include "src/core/NEON/kernels/NECannyEdgeKernel.h"
#include "src/core/NEON/kernels/NEChannelCombineKernel.h"
#include "src/core/NEON/kernels/NEChannelExtractKernel.h"
#include "src/core/NEON/kernels/NEChannelShuffleLayerKernel.h"
#include "src/core/NEON/kernels/NECol2ImKernel.h"
#include "src/core/NEON/kernels/NEColorConvertKernel.h"
#include "src/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEConvolutionKernel.h"
#include "src/core/NEON/kernels/NECropKernel.h"
#include "src/core/NEON/kernels/NECumulativeDistributionKernel.h"
#include "src/core/NEON/kernels/NEDepthConvertLayerKernel.h"
#include "src/core/NEON/kernels/NEDepthToSpaceLayerKernel.h"
#include "src/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"
#include "src/core/NEON/kernels/NEDequantizationLayerKernel.h"
#include "src/core/NEON/kernels/NEDerivativeKernel.h"
#include "src/core/NEON/kernels/NEDilateKernel.h"
#include "src/core/NEON/kernels/NEDirectConvolutionLayerKernel.h"
#include "src/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"
#include "src/core/NEON/kernels/NEErodeKernel.h"
#include "src/core/NEON/kernels/NEFFTDigitReverseKernel.h"
#include "src/core/NEON/kernels/NEFFTRadixStageKernel.h"
#include "src/core/NEON/kernels/NEFFTScaleKernel.h"
#include "src/core/NEON/kernels/NEFastCornersKernel.h"
#include "src/core/NEON/kernels/NEFillArrayKernel.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ScaleKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NEGatherKernel.h"
#include "src/core/NEON/kernels/NEGaussian3x3Kernel.h"
#include "src/core/NEON/kernels/NEGaussian5x5Kernel.h"
#include "src/core/NEON/kernels/NEGaussianPyramidKernel.h"
#include "src/core/NEON/kernels/NEGenerateProposalsLayerKernel.h"
#include "src/core/NEON/kernels/NEHOGDescriptorKernel.h"
#include "src/core/NEON/kernels/NEHOGDetectorKernel.h"
#include "src/core/NEON/kernels/NEHarrisCornersKernel.h"
#include "src/core/NEON/kernels/NEHistogramKernel.h"
#include "src/core/NEON/kernels/NEIm2ColKernel.h"
#include "src/core/NEON/kernels/NEInstanceNormalizationLayerKernel.h"
#include "src/core/NEON/kernels/NEIntegralImageKernel.h"
#include "src/core/NEON/kernels/NEL2NormalizeLayerKernel.h"
#include "src/core/NEON/kernels/NELKTrackerKernel.h"
#include "src/core/NEON/kernels/NELogicalKernel.h"
#include "src/core/NEON/kernels/NEMagnitudePhaseKernel.h"
#include "src/core/NEON/kernels/NEMaxUnpoolingLayerKernel.h"
#include "src/core/NEON/kernels/NEMeanStdDevKernel.h"
#include "src/core/NEON/kernels/NEMeanStdDevNormalizationKernel.h"
#include "src/core/NEON/kernels/NEMedian3x3Kernel.h"
#include "src/core/NEON/kernels/NEMinMaxLayerKernel.h"
#include "src/core/NEON/kernels/NEMinMaxLocationKernel.h"
#include "src/core/NEON/kernels/NENonLinearFilterKernel.h"
#include "src/core/NEON/kernels/NENonMaximaSuppression3x3Kernel.h"
#include "src/core/NEON/kernels/NENormalizationLayerKernel.h"
#include "src/core/NEON/kernels/NEPadLayerKernel.h"
#include "src/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"
#include "src/core/NEON/kernels/NEPriorBoxLayerKernel.h"
#include "src/core/NEON/kernels/NEQLSTMLayerNormalizationKernel.h"
#include "src/core/NEON/kernels/NEQuantizationLayerKernel.h"
#include "src/core/NEON/kernels/NEROIAlignLayerKernel.h"
#include "src/core/NEON/kernels/NEROIPoolingLayerKernel.h"
#include "src/core/NEON/kernels/NERangeKernel.h"
#include "src/core/NEON/kernels/NEReductionOperationKernel.h"
#include "src/core/NEON/kernels/NERemapKernel.h"
#include "src/core/NEON/kernels/NEReorgLayerKernel.h"
#include "src/core/NEON/kernels/NEReverseKernel.h"
#include "src/core/NEON/kernels/NEScaleKernel.h"
#include "src/core/NEON/kernels/NEScharr3x3Kernel.h"
#include "src/core/NEON/kernels/NESelectKernel.h"
#include "src/core/NEON/kernels/NESobel3x3Kernel.h"
#include "src/core/NEON/kernels/NESobel5x5Kernel.h"
#include "src/core/NEON/kernels/NESobel7x7Kernel.h"
#include "src/core/NEON/kernels/NESpaceToBatchLayerKernel.h"
#include "src/core/NEON/kernels/NESpaceToDepthLayerKernel.h"
#include "src/core/NEON/kernels/NEStackLayerKernel.h"
#include "src/core/NEON/kernels/NEStridedSliceKernel.h"
#include "src/core/NEON/kernels/NETableLookupKernel.h"
#include "src/core/NEON/kernels/NEThresholdKernel.h"
#include "src/core/NEON/kernels/NETileKernel.h"
#include "src/core/NEON/kernels/NETransposeKernel.h"
#include "src/core/NEON/kernels/NEWarpKernel.h"
#include "src/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "src/core/NEON/kernels/NEWinogradConvolutionLayerKernel.h"

#endif /* ARM_COMPUTE_NEKERNELS_H */
