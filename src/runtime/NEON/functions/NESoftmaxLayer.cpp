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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cfloat>

using namespace arm_compute;

NESoftmaxLayer::NESoftmaxLayer()
    : _max_kernel(), _shift_exp_sum_kernel(), _norm_kernel(), _fill_border_kernel(), _max(), _sum(), _tmp()
{
}

void NESoftmaxLayer::configure(ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);

    // Create intermediate tensors shapes
    TensorInfo tensor_info_tmp(input->info()->tensor_shape(), input->info()->num_channels(), input->info()->data_type());
    tensor_info_tmp.auto_padding();
    _tmp.allocator()->init(tensor_info_tmp);
    _tmp.allocator()->allocate();

    TensorShape shape = input->info()->tensor_shape();
    shape.set(0, 1);
    TensorInfo tensor_info_max_sum(shape, input->info()->num_channels(), input->info()->data_type());
    tensor_info_max_sum.auto_padding();
    _max.allocator()->init(tensor_info_max_sum);
    _max.allocator()->allocate();
    _sum.allocator()->init(tensor_info_max_sum);
    _sum.allocator()->allocate();

    // Configure Kernels
    _fill_border_kernel.configure(input, 3, BorderMode::CONSTANT, PixelValue(-FLT_MAX));
    _max_kernel.configure(input, &_max);
    _shift_exp_sum_kernel.configure(input, &_max, &_tmp, &_sum);
    _norm_kernel.configure(&_tmp, &_sum, output);
}

void NESoftmaxLayer::run()
{
    NEScheduler::get().multithread(&_fill_border_kernel);
    NEScheduler::get().multithread(&_max_kernel);
    NEScheduler::get().multithread(&_shift_exp_sum_kernel);
    NEScheduler::get().multithread(&_norm_kernel);
}
