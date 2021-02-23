/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/core/TracePoint.h"
#include <stdio.h>
#include <vector>

#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/Pyramid.h"
#include "arm_compute/runtime/common/LSTMParams.h"
#include "src/core/NEON/kernels/assembly/arm_gemm.hpp"
#include "utils/TypePrinter.h"

namespace arm_compute
{
ARM_COMPUTE_TRACE_TO_STRING(KeyPointArray)
ARM_COMPUTE_TRACE_TO_STRING(Pyramid)
ARM_COMPUTE_TRACE_TO_STRING(LSTMParams<ITensor>)
ARM_COMPUTE_TRACE_TO_STRING(FullyConnectedLayerInfo)
ARM_COMPUTE_TRACE_TO_STRING(arm_gemm::Requantize32)

ARM_COMPUTE_CONST_PTR_CLASS(KeyPointArray)
ARM_COMPUTE_CONST_PTR_CLASS(Pyramid)
ARM_COMPUTE_CONST_PTR_CLASS(LSTMParams<ITensor>)
ARM_COMPUTE_CONST_PTR_CLASS(DetectionPostProcessLayerInfo)
ARM_COMPUTE_CONST_PTR_CLASS(FullyConnectedLayerInfo)
ARM_COMPUTE_CONST_PTR_CLASS(GenerateProposalsInfo)
ARM_COMPUTE_CONST_PTR_CLASS(arm_gemm::Requantize32)
} // namespace arm_compute
