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
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/NEON/INEKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
TensorShape get_output_shape(const ITensorInfo *input)
{
    TensorShape  output_shape{ input->tensor_shape() };
    const size_t transpose_w = 16 / input->element_size();
    output_shape.set(0, input->dimension(1) * transpose_w);
    output_shape.set(1, static_cast<size_t>(std::ceil((input->dimension(0) / static_cast<float>(transpose_w)))));
    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), get_output_shape(input));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

void NEGEMMTranspose1xWKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), get_output_shape(input->info()), 1, input->info()->data_type());

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    const size_t vector_size = 16 / input->info()->element_size();

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(vector_size));

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEGEMMTranspose1xWKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));

    return Status{};
}

void NEGEMMTranspose1xWKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    /*
     * Following an example of how the transposition1xW works when the input data type is F32
     *
     *         |a00 a01 a02 a03|
     *         |a10 a11 a12 a13|
     *         |a20 a21 a22 a23| = | a00 a01 a02 a03 || a10 a11 a12 a13 || a20 a21 a22 a23 || a30 a31 a32 a33 |
     *         |a30 a31 a32 a33|
     *
     * The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
     */

    // Set window for output tensor. Set to 0 the X and Y dimensions in order to allow multi-threading implementation and future batched matrix multiplications
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, window);
    Iterator out(_output, win_out);

    const size_t in_width     = _input->info()->dimension(0);
    const size_t element_size = _input->info()->element_size();
    const size_t out_stride   = _output->info()->strides_in_bytes()[1];
    const size_t vector_size  = 16 / element_size;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8_t *in_ptr  = in.ptr();
        uint8_t *const out_ptr = out.ptr() + (id.y() * vector_size) * element_size + (id.x() / vector_size) * out_stride;

        for(size_t k = 0; k < vector_size; ++k)
        {
            // If the input width is not multiple of W, we fill the reference with 0s
            if((id.x() + k) >= in_width)
            {
                std::memset(out_ptr + k * element_size, 0, element_size);
            }
            else
            {
                std::memcpy(out_ptr + k * element_size, in_ptr + k * element_size, element_size);
            }
        }
    },
    in, out);
}
} // namespace arm_compute
