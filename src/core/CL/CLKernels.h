/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLKERNELS_H
#define ARM_COMPUTE_CLKERNELS_H

/* Header regrouping all the CL kernels */
#include "src/core/CL/kernels/CLArgMinMaxLayerKernel.h"
#include "src/core/CL/kernels/CLBatchNormalizationLayerKernel.h"
#include "src/core/CL/kernels/CLBatchToSpaceLayerKernel.h"
#include "src/core/CL/kernels/CLBitwiseKernel.h"
#include "src/core/CL/kernels/CLBoundingBoxTransformKernel.h"
#include "src/core/CL/kernels/CLChannelShuffleLayerKernel.h"
#include "src/core/CL/kernels/CLComparisonKernel.h"
#include "src/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"
#include "src/core/CL/kernels/CLDeconvolutionReshapeOutputKernel.h"
#include "src/core/CL/kernels/CLDepthToSpaceLayerKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"
#include "src/core/CL/kernels/CLFFTDigitReverseKernel.h"
#include "src/core/CL/kernels/CLFFTRadixStageKernel.h"
#include "src/core/CL/kernels/CLFFTScaleKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLFuseBatchNormalizationKernel.h"
#include "src/core/CL/kernels/CLGatherKernel.h"
#include "src/core/CL/kernels/CLGenerateProposalsLayerKernel.h"
#include "src/core/CL/kernels/CLInstanceNormalizationLayerKernel.h"
#include "src/core/CL/kernels/CLL2NormalizeLayerKernel.h"
#include "src/core/CL/kernels/CLMaxUnpoolingLayerKernel.h"
#include "src/core/CL/kernels/CLMeanStdDevNormalizationKernel.h"
#include "src/core/CL/kernels/CLNormalizationLayerKernel.h"
#include "src/core/CL/kernels/CLNormalizePlanarYUVLayerKernel.h"
#include "src/core/CL/kernels/CLPadLayerKernel.h"
#include "src/core/CL/kernels/CLPriorBoxLayerKernel.h"
#include "src/core/CL/kernels/CLQLSTMLayerNormalizationKernel.h"
#include "src/core/CL/kernels/CLROIAlignLayerKernel.h"
#include "src/core/CL/kernels/CLROIPoolingLayerKernel.h"
#include "src/core/CL/kernels/CLRangeKernel.h"
#include "src/core/CL/kernels/CLReductionOperationKernel.h"
#include "src/core/CL/kernels/CLReorgLayerKernel.h"
#include "src/core/CL/kernels/CLReverseKernel.h"
#include "src/core/CL/kernels/CLSelectKernel.h"
#include "src/core/CL/kernels/CLSpaceToBatchLayerKernel.h"
#include "src/core/CL/kernels/CLSpaceToDepthLayerKernel.h"
#include "src/core/CL/kernels/CLStackLayerKernel.h"
#include "src/core/CL/kernels/CLStridedSliceKernel.h"
#include "src/core/CL/kernels/CLTileKernel.h"

#endif /* ARM_COMPUTE_CLKERNELS_H */
