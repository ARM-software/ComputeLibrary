/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuCol2ImKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
using namespace misc::shape_calculator;
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &convolved_dims)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use CPU FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    // Validate configured output
    if (dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           compute_col2im_shape(*src, convolved_dims, false));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}
} // namespace

void CpuCol2ImKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const Size2D &convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, convolved_dims));

    _convolved_dims = convolved_dims;

    // Configure kernel window
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_col2im_shape(*src, convolved_dims, false)));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());

    ICpuKernel::configure(win);
}

Status CpuCol2ImKernel::validate(const ITensorInfo *src, const ITensorInfo *output, const Size2D &convolved_dims)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, output, convolved_dims));
    return Status{};
}

void CpuCol2ImKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    const uint8_t el_size         = src->info()->element_size();
    const int     output_stride_x = dst->info()->strides_in_bytes().x();
    const int     output_stride_y = dst->info()->strides_in_bytes().y();
    const int     output_stride_z = dst->info()->strides_in_bytes().z();

    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Create iterators
    Iterator in(src, window);
    Iterator out(dst, window_out);

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const int hidx = id.y();
            const int idx  = id.x() * output_stride_z + (hidx / _convolved_dims.width) * output_stride_y +
                            (hidx % _convolved_dims.width) * output_stride_x;
            std::memcpy(out.ptr() + idx, in.ptr(), el_size);
        },
        in, out);
}

const char *CpuCol2ImKernel::name() const
{
    return "CpuCol2ImKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
