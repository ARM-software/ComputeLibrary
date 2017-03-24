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
#include "arm_compute/runtime/NEON/functions/NEFastCorners.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"

using namespace arm_compute;

NEFastCorners::NEFastCorners()
    : _fast_corners_kernel(),
      _border_handler(),
      _nonmax_kernel(),
      _fill_kernel(),
      _out_border_handler_kernel(),
      _output(),
      _suppressed(),
      _non_max(false)
{
}

void NEFastCorners::configure(IImage *input, float threshold, bool nonmax_suppression, KeyPointArray *const corners,
                              BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(BorderMode::UNDEFINED != border_mode);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == corners);
    ARM_COMPUTE_ERROR_ON(threshold < 1 && threshold > 255);

    TensorInfo tensor_info(input->info()->tensor_shape(), Format::U8);
    _output.allocator()->init(tensor_info);
    _border_handler.configure(input, _fast_corners_kernel.border_size(), border_mode, constant_border_value);
    /*
        If border is UNDEFINED _fast_corners_kernel will operate in xwindow (3, width - 3) and ywindow (3, height -3) so
        the output image will leave the pixels on the borders unchanged. This can cause problems if Non Max Suppression is performed afterwards.

        If non max sup is true && border == UNDEFINED we must set the border texels to 0 before executing the non max sup kernel
    */
    _fast_corners_kernel.configure(input, &_output, threshold, nonmax_suppression, BorderMode::UNDEFINED == border_mode);

    _output.allocator()->allocate();
    _non_max = nonmax_suppression;

    if(!_non_max)
    {
        _fill_kernel.configure(&_output, 1 /* we keep all texels >0 */, corners);
    }
    else
    {
        if(border_mode == BorderMode::UNDEFINED)
        {
            // We use this kernel to set the borders to 0 before performing non max sup
            _out_border_handler_kernel.configure(&_output, _fast_corners_kernel.border_size(), PixelValue(static_cast<uint8_t>(0)));
        }

        _suppressed.allocator()->init(tensor_info);
        _suppressed.allocator()->allocate();
        _nonmax_kernel.configure(&_output, &_suppressed, BorderMode::UNDEFINED == border_mode);
        _fill_kernel.configure(&_suppressed, 1 /* we keep all texels >0 */, corners);
    }
}

void NEFastCorners::run()
{
    NEScheduler::get().multithread(&_fast_corners_kernel);

    if(_non_max)
    {
        NEScheduler::get().multithread(&_out_border_handler_kernel); // make sure inner borders are set to 0 before running non max sup kernel
        NEScheduler::get().multithread(&_nonmax_kernel);
    }

    NEScheduler::get().multithread(&_fill_kernel);
}
