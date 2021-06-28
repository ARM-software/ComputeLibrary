/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLBitwiseKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
CLBitwiseKernel::CLBitwiseKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLBitwiseKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, BitwiseOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8);
    if(op != BitwiseOperation::NOT)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input2);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8);
    }
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*(output->info()), *(input1->info()));
    auto padding_info = get_padding_info({ input1, input2, output });

    // Configure kernel window
    const unsigned int vec_size_x = adjust_vec_size(16 / output->info()->element_size(), output->info()->dimension(0));
    Window             win        = calculate_max_window(*output->info(), Steps(vec_size_x));

    _input1 = input1;
    _input2 = input2;
    _output = output;

    // Create kernel
    std::string kernel_name = "";
    switch(op)
    {
        case BitwiseOperation::AND:
            kernel_name = "bitwise_and";
            break;
        case BitwiseOperation::NOT:
            kernel_name = "bitwise_not";
            break;
        case BitwiseOperation::OR:
            kernel_name = "bitwise_or";
            break;
        case BitwiseOperation::XOR:
            kernel_name = "bitwise_xor";
            break;
        default:
            ARM_COMPUTE_ERROR("Bitwise operation not supported");
    }

    CLBuildOptions build_opts;
    const int      vec_size_x_leftovers = output->info()->dimension(0) % vec_size_x;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftovers));
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    ICLKernel::configure_internal(win);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void CLBitwiseKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input1, slice);
        if(_input2 != nullptr)
        {
            add_2D_tensor_argument(idx, _input2, slice);
        }
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
} // namespace arm_compute