/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLKERNELS_H__
#define __ARM_COMPUTE_CLKERNELS_H__

/* Header regrouping all the CL kernels */
#include "arm_compute/core/CL/kernels/CLAbsoluteDifferenceKernel.h"
#include "arm_compute/core/CL/kernels/CLAccumulateKernel.h"
#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLArithmeticAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLArithmeticSubtractionKernel.h"
#include "arm_compute/core/CL/kernels/CLBatchNormalizationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLBitwiseAndKernel.h"
#include "arm_compute/core/CL/kernels/CLBitwiseNotKernel.h"
#include "arm_compute/core/CL/kernels/CLBitwiseOrKernel.h"
#include "arm_compute/core/CL/kernels/CLBitwiseXorKernel.h"
#include "arm_compute/core/CL/kernels/CLBox3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLCannyEdgeKernel.h"
#include "arm_compute/core/CL/kernels/CLChannelCombineKernel.h"
#include "arm_compute/core/CL/kernels/CLChannelExtractKernel.h"
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"
#include "arm_compute/core/CL/kernels/CLColorConvertKernel.h"
#include "arm_compute/core/CL/kernels/CLConvolutionKernel.h"
#include "arm_compute/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthConcatenateLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthConvertLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseVectorToTensorKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseWeightsReshapeKernel.h"
#include "arm_compute/core/CL/kernels/CLDequantizationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLDerivativeKernel.h"
#include "arm_compute/core/CL/kernels/CLDilateKernel.h"
#include "arm_compute/core/CL/kernels/CLDirectConvolutionLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLErodeKernel.h"
#include "arm_compute/core/CL/kernels/CLFastCornersKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLFloorKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ToUint8ScaleKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixVectorMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/core/CL/kernels/CLGaussian3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLGaussian5x5Kernel.h"
#include "arm_compute/core/CL/kernels/CLGaussianPyramidKernel.h"
#include "arm_compute/core/CL/kernels/CLHOGDescriptorKernel.h"
#include "arm_compute/core/CL/kernels/CLHOGDetectorKernel.h"
#include "arm_compute/core/CL/kernels/CLHarrisCornersKernel.h"
#include "arm_compute/core/CL/kernels/CLHistogramKernel.h"
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLIntegralImageKernel.h"
#include "arm_compute/core/CL/kernels/CLL2NormalizeLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLLKTrackerKernel.h"
#include "arm_compute/core/CL/kernels/CLLocallyConnectedMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLMagnitudePhaseKernel.h"
#include "arm_compute/core/CL/kernels/CLMeanStdDevKernel.h"
#include "arm_compute/core/CL/kernels/CLMedian3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLMinMaxLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLMinMaxLocationKernel.h"
#include "arm_compute/core/CL/kernels/CLNonLinearFilterKernel.h"
#include "arm_compute/core/CL/kernels/CLNonMaximaSuppression3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLNormalizationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLPermuteKernel.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/CL/kernels/CLPoolingLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLQuantizationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLROIPoolingLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"
#include "arm_compute/core/CL/kernels/CLRemapKernel.h"
#include "arm_compute/core/CL/kernels/CLReshapeLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLScaleKernel.h"
#include "arm_compute/core/CL/kernels/CLScharr3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLSobel3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLSobel5x5Kernel.h"
#include "arm_compute/core/CL/kernels/CLSobel7x7Kernel.h"
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLTableLookupKernel.h"
#include "arm_compute/core/CL/kernels/CLThresholdKernel.h"
#include "arm_compute/core/CL/kernels/CLTransposeKernel.h"
#include "arm_compute/core/CL/kernels/CLWarpAffineKernel.h"
#include "arm_compute/core/CL/kernels/CLWarpPerspectiveKernel.h"
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"

#endif /* __ARM_COMPUTE_CLKERNELS_H__ */
