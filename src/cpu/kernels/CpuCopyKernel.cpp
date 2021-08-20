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
#include "src/cpu/kernels/CpuCopyKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const PaddingList &padding = PaddingList())
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > 4);

    // Validate destination if initialized
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(misc::shape_calculator::compute_padded_shape(src->tensor_shape(), padding), dst->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *src, ITensorInfo *dst)
{
    // Destination auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, *src);
    return std::make_pair(Status{}, calculate_max_window(*dst));
}

std::pair<Status, Window> validate_and_configure_window_with_padding(const ITensorInfo *src, ITensorInfo *dst, const PaddingList &padding)
{
    const TensorShape src_shape    = src->tensor_shape();
    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(src_shape, padding);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(padded_shape));
    // Configure window
    const Window win = calculate_max_window(*dst, dst->dimension(0));
    return std::make_pair(Status{}, win);
}

} // namespace

void CpuCopyKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const PaddingList &padding)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, padding));

    _padding = padding;

    std::pair<Status, Window> win_config;
    if(padding.empty())
    {
        win_config = validate_and_configure_window(src, dst);
    }
    else
    {
        win_config = validate_and_configure_window_with_padding(src, dst, padding);
    }

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuCopyKernel::validate(const arm_compute::ITensorInfo *src, const arm_compute::ITensorInfo *dst, const PaddingList &padding)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, padding));

    if(padding.empty())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get()).first);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_with_padding(src->clone().get(), dst->clone().get(), padding).first);
    }

    return Status{};
}

void CpuCopyKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    if(_padding.empty())
    {
        Window dst_window{ window };
        dst_window.set(Window::DimX, Window::Dimension(dst_window.x().start(), dst_window.x().end(), src->info()->dimension(0)));
        Window out_slice = dst_window.first_slice_window_1D();
        do
        {
            Iterator src_it(src, out_slice);
            Iterator dst_it(dst, out_slice);

            execute_window_loop(out_slice, [&](const Coordinates &)
            {
                memcpy(dst_it.ptr(), src_it.ptr(), dst->info()->dimension(0) * dst->info()->element_size());
            },
            src_it, dst_it);
        }
        while(dst_window.slide_window_slice_1D(out_slice));
    }
    else
    {
        Window src_window{ window };
        src_window.set(Window::DimX, Window::Dimension(0, window.x().end() - _padding[0].first, src->info()->dimension(0)));

        Iterator     src_it(src, src_window);
        Iterator     dst_it(dst, window);
        const size_t row_size_in_bytes = src->info()->dimension(0) * src->info()->element_size();
        execute_window_loop(window, [&](const Coordinates &)
        {
            auto dst_ptr = dst_it.ptr() + _padding[0].first * dst->info()->element_size();
            std::memcpy(dst_ptr, src_it.ptr(), row_size_in_bytes);
        },
        src_it, dst_it);
    }
}

const char *CpuCopyKernel::name() const
{
    return "CpuCopyKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
