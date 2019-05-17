/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCFUNCTIONS_H__
#define __ARM_COMPUTE_GCFUNCTIONS_H__

/* Header regrouping all the GLES compute functions */
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCAbsoluteDifference.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCArithmeticAddition.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCBatchNormalizationLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConcatenateLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConvolutionLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDirectConvolutionLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDropoutLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCFillBorder.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCFullyConnectedLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMM.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMMInterleave4x4.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMMTranspose1xW.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCNormalizationLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCNormalizePlanarYUVLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCPixelWiseMultiplication.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCPoolingLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCScale.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCSoftmaxLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCTensorShift.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCTranspose.h"

#endif /* __ARM_COMPUTE_GCFUNCTIONS_H__ */
