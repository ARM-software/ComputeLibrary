/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCKERNELS_H__
#define __ARM_COMPUTE_GCKERNELS_H__

/* Header regrouping all the GLES compute kernels */
#include "arm_compute/core/GLES_COMPUTE/kernels/GCAbsoluteDifferenceKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCActivationLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCArithmeticAdditionKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCBatchNormalizationLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCCol2ImKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthConcatenateLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDirectConvolutionLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDropoutLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMTranspose1xWKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizationLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizePlanarYUVLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCPoolingLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCScaleKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCSoftmaxLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTransposeKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCWeightsReshapeKernel.h"

#endif /* __ARM_COMPUTE_GCKERNELS_H__ */
