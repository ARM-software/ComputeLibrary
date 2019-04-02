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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCSoftmaxLayer.h"

#include "arm_compute/core/GLES_COMPUTE/kernels/GCSoftmaxLayerKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

using namespace arm_compute;

GCSoftmaxLayer::GCSoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_kernel(), _shift_exp_sum_kernel(), _norm_kernel(), _max(), _sum(), _tmp()
{
}

void GCSoftmaxLayer::configure(const IGCTensor *input, IGCTensor *output, float beta, size_t axis)
{
    ARM_COMPUTE_UNUSED(beta, axis);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(beta != 1.0f);
    ARM_COMPUTE_ERROR_ON_MSG(axis != 1, "Axis must be 1 for GLES");

    // Create intermediate tensors shapes
    _tmp.allocator()->init(TensorInfo(input->info()->tensor_shape(), input->info()->num_channels(), input->info()->data_type()));

    TensorShape shape = input->info()->tensor_shape();
    shape.set(0, 1);
    TensorInfo tensor_info_max_sum(shape, input->info()->num_channels(), input->info()->data_type());
    _max.allocator()->init(tensor_info_max_sum);
    _sum.allocator()->init(tensor_info_max_sum);

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);
    _memory_group.manage(&_max);
    _memory_group.manage(&_sum);

    // Configure Kernels
    _max_kernel.configure(input, &_max);
    _shift_exp_sum_kernel.configure(input, &_max, &_tmp, &_sum);
    _norm_kernel.configure(&_tmp, &_sum, output);

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
}

void GCSoftmaxLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    GCScheduler::get().dispatch(_max_kernel, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_shift_exp_sum_kernel, false);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_norm_kernel);
}
