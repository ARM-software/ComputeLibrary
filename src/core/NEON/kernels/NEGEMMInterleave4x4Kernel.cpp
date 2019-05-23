/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S8,
                                                         DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    if(output->total_size() != 0)
    {
        TensorShape output_shape = input->tensor_shape();
        output_shape.set(0, input->dimension(0) * 4);
        output_shape.set(1, std::ceil(input->dimension(1) / 4.0f));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    unsigned int           num_elems_processed_per_iteration_x = (input->element_size() == 1) ? 8 : 4;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;
    bool                   window_changed                      = false;

    // Configure kernel window
    Window                win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    window_changed = window_changed || update_window_and_padding(win, input_access);

    // Configure window in case of configured output
    if(output->total_size() != 0)
    {
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y, 1, 4.0f, 0.25f);
        window_changed = window_changed || update_window_and_padding(win, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

void gemm_interleave_8bit_elements(const ITensor *input, ITensor *output, const Window &window)
{
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    // Set window for output tensor
    Window win_out(window);
    win_out.scale(Window::DimY, 0.25f);
    Iterator in(input, window);

    win_out.set_dimension_step(Window::DimX, 32);
    Iterator out(output, win_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x8x4_t data =
        {
            {
                vld1_u8(in.ptr() + 0 * in_stride),
                vld1_u8(in.ptr() + 1 * in_stride),
                vld1_u8(in.ptr() + 2 * in_stride),
                vld1_u8(in.ptr() + 3 * in_stride),
            }
        };
        vst4_u8(out.ptr(), data);
    },
    in, out);
}

void gemm_interleave_16bit_elements(const ITensor *input, ITensor *output, const Window &window)
{
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    // Set window for output tensor
    Window win_out(window);
    win_out.scale(Window::DimY, 0.25f);
    Iterator in(input, window);

    win_out.set_dimension_step(Window::DimX, 16);
    Iterator out(output, win_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint16x4x4_t data =
        {
            {
                vld1_u16(reinterpret_cast<const uint16_t *>(in.ptr() + 0 * in_stride)),
                vld1_u16(reinterpret_cast<const uint16_t *>(in.ptr() + 1 * in_stride)),
                vld1_u16(reinterpret_cast<const uint16_t *>(in.ptr() + 2 * in_stride)),
                vld1_u16(reinterpret_cast<const uint16_t *>(in.ptr() + 3 * in_stride)),
            }
        };
        vst4_u16(reinterpret_cast<uint16_t *>(out.ptr()), data);
    },
    in, out);
}

void gemm_interleave_32bit_elements(const ITensor *input, ITensor *output, const Window &window)
{
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    // Set window for output tensor
    Window win_out(window);
    win_out.scale(Window::DimY, 0.25f);
    Iterator in(input, window);

    win_out.set_dimension_step(Window::DimX, 16);
    Iterator out(output, win_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint32x4x4_t data =
        {
            {
                vld1q_u32(reinterpret_cast<const uint32_t *>(in.ptr() + 0 * in_stride)),
                vld1q_u32(reinterpret_cast<const uint32_t *>(in.ptr() + 1 * in_stride)),
                vld1q_u32(reinterpret_cast<const uint32_t *>(in.ptr() + 2 * in_stride)),
                vld1q_u32(reinterpret_cast<const uint32_t *>(in.ptr() + 3 * in_stride))
            }
        };
        vst4q_u32(reinterpret_cast<uint32_t *>(out.ptr()), data);
    },
    in, out);
}
} // namespace

NEGEMMInterleave4x4Kernel::NEGEMMInterleave4x4Kernel()
    : _func(nullptr)
{
}

void NEGEMMInterleave4x4Kernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_interleaved_shape(*input->info())));

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    switch(input->info()->element_size())
    {
        case 1:
            _func = &gemm_interleave_8bit_elements;
            break;
        case 2:
            _func = &gemm_interleave_16bit_elements;
            break;
        case 4:
            _func = &gemm_interleave_32bit_elements;
            break;
        default:
            ARM_COMPUTE_ERROR_ON("Element size not supported");
            break;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMInterleave4x4Kernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEGEMMInterleave4x4Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    /*
    *  This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
    *         |a00 a01 a02 a03|
    *         |a10 a11 a12 a13|
    *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
    *         |a30 a31 a32 a33|
    *
    *         After this operation, the output matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
    */
    (*_func)(_input, _output, window);
}
