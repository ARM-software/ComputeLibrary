/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPPermute.h"

#include "arm_compute/core/CPP/kernels/CPPPermuteKernel.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"

using namespace arm_compute;

void CPPPermute::configure(const ITensor *input, ITensor *output, const PermutationVector &perm)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, perm);

    auto k = std::make_unique<CPPPermuteKernel>();
    k->configure(input, output, perm);
    _kernel = std::move(k);
}

Status CPPPermute::validate(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);
    return CPPPermuteKernel::validate(input, output, perm);
}
