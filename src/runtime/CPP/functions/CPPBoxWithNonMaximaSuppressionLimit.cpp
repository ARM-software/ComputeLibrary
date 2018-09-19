/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPBoxWithNonMaximaSuppressionLimit.h"

#include "arm_compute/core/CPP/kernels/CPPBoxWithNonMaximaSuppressionLimitKernel.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void CPPBoxWithNonMaximaSuppressionLimit::configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                                                    ITensor *batch_splits_out, ITensor *keeps, ITensor *keeps_size, const BoxNMSLimitInfo info)
{
    auto k = arm_compute::support::cpp14::make_unique<CPPBoxWithNonMaximaSuppressionLimitKernel>();
    k->configure(scores_in, boxes_in, batch_splits_in, scores_out, boxes_out, classes, batch_splits_out, keeps, keeps_size, info);
    _kernel = std::move(k);
}