/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGaussian5x5Kernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

#include <utility>

using namespace arm_compute;

CLGaussian5x5::CLGaussian5x5(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _kernel_hor(), _kernel_vert(), _border_handler(), _tmp()
{
}

void CLGaussian5x5::configure(ICLTensor *input, ICLTensor *output, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);

    _tmp.allocator()->init(TensorInfo(input->info()->tensor_shape(), 1, DataType::U16));

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);

    // Configure kernels
    _kernel_hor.configure(input, &_tmp, border_mode == BorderMode::UNDEFINED);
    _kernel_vert.configure(&_tmp, output, border_mode == BorderMode::UNDEFINED);
    _border_handler.configure(input, _kernel_hor.border_size(), border_mode, PixelValue(constant_border_value));

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
}

void CLGaussian5x5::run()
{
    CLScheduler::get().enqueue(_border_handler, false);

    _memory_group.acquire();

    CLScheduler::get().enqueue(_kernel_hor, false);
    CLScheduler::get().enqueue(_kernel_vert);

    _memory_group.release();
}
