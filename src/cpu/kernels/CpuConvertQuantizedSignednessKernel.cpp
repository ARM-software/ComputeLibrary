/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuConvertQuantizedSignednessKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);

    // Validate output if initialized
    if (dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(src->tensor_shape(), dst->tensor_shape());
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *src, ITensorInfo *dst)
{
    // Output auto inizialitation if not yet initialized
    {
        const bool                    is_input_signed = src->data_type() == DataType::QASYMM8_SIGNED;
        const DataType                dt              = is_input_signed ? DataType::QASYMM8 : DataType::QASYMM8_SIGNED;
        const UniformQuantizationInfo qinfo           = src->quantization_info().uniform();
        const int                     offset_correction = is_input_signed ? -128 : 128;
        const QuantizationInfo        corrected_qinfo = QuantizationInfo(qinfo.scale, qinfo.offset + offset_correction);

        auto_init_if_empty(*dst, src->clone()->set_data_type(dt).set_quantization_info(corrected_qinfo));
    }

    return std::make_pair(Status{}, calculate_max_window(*dst));
}
} // namespace

void CpuConvertQuantizedSignednessKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    std::pair<Status, Window> win_config = validate_and_configure_window(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuConvertQuantizedSignednessKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void CpuConvertQuantizedSignednessKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const uint8_t mask  = 128;
    const auto    vmask = wrapper::vdup_n(mask, wrapper::traits::vector_128_tag{});

    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin = wrapper::vloadq(input_ptr + x);
                wrapper::vstore(output_ptr + x, wrapper::veor(vin, vmask));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const uint8_t in  = *(reinterpret_cast<const uint8_t *>(input_ptr + x));
                *(output_ptr + x) = in ^ mask;
            }
        },
        input, output);
}

const char *CpuConvertQuantizedSignednessKernel::name() const
{
    return "CpuConvertQuantizedSignednessKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
