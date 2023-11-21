/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

void CpuGemmInterleave4x4Kernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_interleaved_shape(*src)));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmInterleave4x4Kernel::validate(src, dst));

    Window win = calculate_max_window(*src, Steps(1, 4));
    ICPPKernel::configure(win);
}

Status CpuGemmInterleave4x4Kernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use CPU FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if (dst->total_size() != 0)
    {
        const TensorShape dst_shape = compute_interleaved_shape(*src);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}

void CpuGemmInterleave4x4Kernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    /*
    *  This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
    *         |a00 a01 a02 a03|
    *         |a10 a11 a12 a13|
    *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
    *         |a30 a31 a32 a33|
    *
    *         After this operation, the dst matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
    */
    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    const size_t window_start_x = window.x().start();
    const size_t window_end_x   = window.x().end();

    const size_t in_height = src->info()->dimension(1);
    const size_t in_stride = src->info()->strides_in_bytes()[1];

    const size_t partial_y = in_height % 4;

    const size_t element_size = src->info()->element_size();

    // Set window for the src tensor
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Set window for the dst tensor
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 1, 1));
    win_out.scale(Window::DimY, 0.25f);

    Iterator in(src, win);
    Iterator out(dst, win_out);

    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            if (id.y() + 4 <= static_cast<int>(in_height))
            {
                for (size_t x = window_start_x; x < window_end_x; ++x)
                {
                    std::memcpy(out.ptr() + (x * 4 + 0) * element_size, (in.ptr() + 0 * in_stride) + x * element_size,
                                element_size);
                    std::memcpy(out.ptr() + (x * 4 + 1) * element_size, (in.ptr() + 1 * in_stride) + x * element_size,
                                element_size);
                    std::memcpy(out.ptr() + (x * 4 + 2) * element_size, (in.ptr() + 2 * in_stride) + x * element_size,
                                element_size);
                    std::memcpy(out.ptr() + (x * 4 + 3) * element_size, (in.ptr() + 3 * in_stride) + x * element_size,
                                element_size);
                }
            }
            else
            {
                for (size_t x = window_start_x; x < window_end_x; ++x)
                {
                    size_t y = 0;
                    for (; y < partial_y; ++y)
                    {
                        std::memcpy(out.ptr() + (x * 4 + y) * element_size,
                                    (in.ptr() + y * in_stride) + x * element_size, element_size);
                    }
                    for (; y < 4; ++y)
                    {
                        std::memset(out.ptr() + (x * 4 + y) * element_size, 0, element_size);
                    }
                }
            }
        },
        in, out);
}

const char *CpuGemmInterleave4x4Kernel::name() const
{
    return "CpuGemmInterleave4x4Kernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
