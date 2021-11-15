/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuFillKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
void CpuFillKernel::configure(const ITensorInfo *tensor, const PixelValue &constant_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    _constant_value = constant_value;

    // Configure kernel window
    Window win = calculate_max_window(*tensor, Steps());
    ICpuKernel::configure(win);
}

void CpuFillKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto inout = tensors.get_tensor(TensorType::ACL_SRC_DST);

    // Collapse all the batches on the third dimension
    bool   has_collapsed = true;
    Window collapsed     = window.collapse_if_possible(window, Window::DimZ, &has_collapsed);
    ARM_COMPUTE_ERROR_ON(!has_collapsed);

    uint8_t *const start_valid_region = inout->ptr_to_element(inout->info()->valid_region().anchor);
    const auto     window_width       = static_cast<int>(collapsed.x().end()) - static_cast<int>(collapsed.x().start());
    const size_t   element_size       = inout->info()->element_size();

    // Unroll X dimension
    collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator tensor_it(inout, collapsed);
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

const char *CpuFillKernel::name() const
{
    return "CpuFillKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
