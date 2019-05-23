/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEMemsetKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
NEMemsetKernel::NEMemsetKernel()
    : _tensor(nullptr), _constant_value()
{
}

void NEMemsetKernel::configure(ITensor *tensor, const PixelValue &constant_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    _tensor         = tensor;
    _constant_value = constant_value;

    // Configure kernel window
    Window win = calculate_max_window(*tensor->info(), Steps());
    INEKernel::configure(win);
}

void NEMemsetKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Collapse all the batches on the third dimension
    bool   has_collapsed = true;
    Window collapsed     = window.collapse_if_possible(window, Window::DimZ, &has_collapsed);
    ARM_COMPUTE_ERROR_ON(!has_collapsed);

    uint8_t *const start_valid_region = _tensor->ptr_to_element(_tensor->info()->valid_region().anchor);
    const auto     window_width       = static_cast<int>(collapsed.x().end()) - static_cast<int>(collapsed.x().start());
    const size_t   element_size       = _tensor->info()->element_size();

    // Unroll X dimension
    collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator tensor_it(_tensor, collapsed);
    execute_window_loop(collapsed, [&](const Coordinates &)
    {
        uint8_t *base_addr = start_valid_region + tensor_it.offset();
        // Set memory
        for(int i = 0; i < window_width; ++i)
        {
            std::memcpy(base_addr + i * element_size, &_constant_value.value, element_size);
        }

    },
    tensor_it);
}
} // namespace arm_compute
