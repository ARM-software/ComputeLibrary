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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void NEGEMMLowpQuantizeDownInt32ToUint8Scale::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_offset, int result_mult_int, int result_shift, int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel>();
    k->configure(input, bias, output, result_offset, result_mult_int, result_shift, min, max);
    _kernel = std::move(k);
}

Status NEGEMMLowpQuantizeDownInt32ToUint8Scale::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    return NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::validate(input, bias, output, min, max);
}

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift,
                                                                    int result_offset_after_shift, int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
    _kernel = std::move(k);
}

Status NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    return NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}