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
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"

using namespace arm_compute;

NESobel7x7::NESobel7x7(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _sobel_hor(), _sobel_vert(), _tmp_x(), _tmp_y(), _border_handler()
{
}

void NESobel7x7::configure(ITensor *input, ITensor *output_x, ITensor *output_y, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);

    const bool run_sobel_x = output_x != nullptr;
    const bool run_sobel_y = output_y != nullptr;

    TensorInfo tensor_info(input->info()->tensor_shape(), Format::S32);

    if(run_sobel_x && run_sobel_y)
    {
        _tmp_x.allocator()->init(tensor_info);
        _tmp_y.allocator()->init(tensor_info);
        _memory_group.manage(&_tmp_x);
        _memory_group.manage(&_tmp_y);
        _sobel_hor.configure(input, &_tmp_x, &_tmp_y, border_mode == BorderMode::UNDEFINED);
        _sobel_vert.configure(&_tmp_x, &_tmp_y, output_x, output_y, border_mode == BorderMode::UNDEFINED);
        _tmp_x.allocator()->allocate();
        _tmp_y.allocator()->allocate();
    }
    else if(run_sobel_x)
    {
        _tmp_x.allocator()->init(tensor_info);
        _memory_group.manage(&_tmp_x);
        _sobel_hor.configure(input, &_tmp_x, nullptr, border_mode == BorderMode::UNDEFINED);
        _sobel_vert.configure(&_tmp_x, nullptr, output_x, nullptr, border_mode == BorderMode::UNDEFINED);
        _tmp_x.allocator()->allocate();
    }
    else if(run_sobel_y)
    {
        _tmp_y.allocator()->init(tensor_info);
        _memory_group.manage(&_tmp_y);
        _sobel_hor.configure(input, nullptr, &_tmp_y, border_mode == BorderMode::UNDEFINED);
        _sobel_vert.configure(nullptr, &_tmp_y, nullptr, output_y, border_mode == BorderMode::UNDEFINED);
        _tmp_y.allocator()->allocate();
    }

    _border_handler.configure(input, _sobel_hor.border_size(), border_mode, PixelValue(constant_border_value));
}

void NESobel7x7::run()
{
    NEScheduler::get().schedule(&_border_handler, Window::DimZ);

    _memory_group.acquire();

    NEScheduler::get().schedule(&_sobel_hor, Window::DimY);
    NEScheduler::get().schedule(&_sobel_vert, Window::DimY);

    _memory_group.release();
}
