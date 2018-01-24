/*
 * Copyright (c) 2017 ARM Limited.
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

#include "arm_compute/runtime/GLES_COMPUTE/functions/GCNormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

using namespace arm_compute;

GCNormalizationLayer::GCNormalizationLayer()
    : _squared_input(), _norm_kernel(), _multiply_kernel(), _border_handler()
{
}

void GCNormalizationLayer::configure(const IGCTensor *input, IGCTensor *output, const NormalizationLayerInfo &norm_info)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);

    _squared_input.allocator()->init(TensorInfo(input->info()->tensor_shape(), 1, input->info()->data_type()));

    _norm_kernel.configure(input, &_squared_input, output, norm_info);
    _multiply_kernel.configure(input, input, &_squared_input, 1.0f);
    // Fill the border by 3 elements since we need vload4 in the IN_MAP normalization kernel
    _border_handler.configure(&_squared_input, _norm_kernel.border_size(), BorderMode::CONSTANT, PixelValue(0));

    // Allocate intermediate buffers
    _squared_input.allocator()->allocate();
}

void GCNormalizationLayer::run()
{
    GCScheduler::get().dispatch(_multiply_kernel, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_border_handler, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_norm_kernel, true);
}
