/*
 * Copyright (c) 2019 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLDepthToSpaceLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLDepthToSpaceLayer::CLDepthToSpaceLayer()
    : _depth_to_space_kernel()
{
}

void CLDepthToSpaceLayer::configure(const ICLTensor *input, ICLTensor *output, int32_t block_shape)
{
    _depth_to_space_kernel.configure(input, output, block_shape);
}

Status CLDepthToSpaceLayer::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    return CLDepthToSpaceLayerKernel::validate(input, output, block_shape);
}

void CLDepthToSpaceLayer::run()
{
    CLScheduler::get().enqueue(_depth_to_space_kernel, true);
}
