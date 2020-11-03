/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGaussian5x5.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/NEON/kernels/NEGaussian5x5Kernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEGaussian5x5::~NEGaussian5x5() = default;

NEGaussian5x5::NEGaussian5x5(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _kernel_hor(), _kernel_vert(), _tmp(), _border_handler()
{
}

void NEGaussian5x5::configure(ITensor *input, ITensor *output, BorderMode border_mode, uint8_t constant_border_value)
{
    // Init temporary buffer
    TensorInfo tensor_info(input->info()->tensor_shape(), 1, DataType::S16);
    _tmp.allocator()->init(tensor_info);

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);

    _kernel_hor     = arm_compute::support::cpp14::make_unique<NEGaussian5x5HorKernel>();
    _kernel_vert    = arm_compute::support::cpp14::make_unique<NEGaussian5x5VertKernel>();
    _border_handler = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();

    // Create and configure kernels for the two passes
    _kernel_hor->configure(input, &_tmp, border_mode == BorderMode::UNDEFINED);
    _kernel_vert->configure(&_tmp, output, border_mode == BorderMode::UNDEFINED);

    _tmp.allocator()->allocate();

    _border_handler->configure(input, _kernel_hor->border_size(), border_mode, PixelValue(constant_border_value));
}

void NEGaussian5x5::run()
{
    NEScheduler::get().schedule(_border_handler.get(), Window::DimZ);

    MemoryGroupResourceScope scope_mg(_memory_group);

    NEScheduler::get().schedule(_kernel_hor.get(), Window::DimY);
    NEScheduler::get().schedule(_kernel_vert.get(), Window::DimY);
}
} // namespace arm_compute