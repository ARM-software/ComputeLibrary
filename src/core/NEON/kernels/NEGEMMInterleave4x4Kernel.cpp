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
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/NEON/INEKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

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
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
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
            _func = &NEGEMMInterleave4x4Kernel::gemm_interleave4x4<uint8_t>;
            break;
        case 2:
            _func = &NEGEMMInterleave4x4Kernel::gemm_interleave4x4<uint16_t>;
            break;
        case 4:
            _func = &NEGEMMInterleave4x4Kernel::gemm_interleave4x4<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR_ON("Element size not supported");
            break;
    }

    Window win = calculate_max_window(*input->info(), Steps(1, 4));

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEGEMMInterleave4x4Kernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));

    return Status{};
}

template <typename ScalarType>
void NEGEMMInterleave4x4Kernel::gemm_interleave4x4(const ITensor *input, ITensor *output, const Window &window)
{
    const size_t window_start_x = window.x().start();
    const size_t window_end_x   = window.x().end();

    const size_t in_height = input->info()->dimension(1);
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    const size_t partial_y = in_height % 4;

    // Set window for the input tensor
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Set window for the output tensor
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 1, 1));
    win_out.scale(Window::DimY, 0.25f);

    Iterator in(input, win);
    Iterator out(output, win_out);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        if(id.y() + 4 <= static_cast<int>(in_height))
        {
            for(size_t x = window_start_x; x < window_end_x; ++x)
            {
                const ScalarType data[4] =
                {
                    *(reinterpret_cast<const ScalarType *>(in.ptr() + 0 * in_stride) + x),
                    *(reinterpret_cast<const ScalarType *>(in.ptr() + 1 * in_stride) + x),
                    *(reinterpret_cast<const ScalarType *>(in.ptr() + 2 * in_stride) + x),
                    *(reinterpret_cast<const ScalarType *>(in.ptr() + 3 * in_stride) + x),
                };
                std::memcpy(out.ptr() + x * 4 * sizeof(ScalarType), data, 4 * sizeof(ScalarType));
            }
        }
        else
        {
            for(size_t x = window_start_x; x < window_end_x; ++x)
            {
                ScalarType data[4] = { 0, 0, 0, 0 };

                for(size_t y = 0; y < partial_y; ++y)
                {
                    data[y] = *(reinterpret_cast<const ScalarType *>(in.ptr() + y * in_stride) + x);
                }

                std::memcpy(out.ptr() + x * 4 * sizeof(ScalarType), data, 4 * sizeof(ScalarType));
            }
        }
    },
    in, out);
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
    (this->*_func)(_input, _output, window);
}
