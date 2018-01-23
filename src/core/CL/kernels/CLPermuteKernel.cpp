/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLPermuteKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLPermuteKernel::CLPermuteKernel()
    : _input(nullptr), _output(nullptr), _perm()
{
}
namespace
{
TensorShape get_output_shape(const ITensorInfo *input, const PermutationVector &perm)
{
    TensorShape output_shape = input->tensor_shape();
    permute(output_shape, perm);
    return output_shape;
}
} // namespace

void CLPermuteKernel::configure(const ICLTensor *input, ICLTensor *output, const PermutationVector &perm)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::QASYMM8,
                                                  DataType::U16, DataType::S16, DataType::QS16,
                                                  DataType::U32, DataType::S32,
                                                  DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MSG(input->info()->num_dimensions() < 3, "Invalid input size!");
    ARM_COMPUTE_ERROR_ON_MSG(
        (perm.num_dimensions() != 3 && ((perm[0] != 2 && perm[1] != 0 && perm[2] != 1) || (perm[0] != 1 && perm[1] != 2 && perm[2] != 0))) && (perm.num_dimensions() != 4 && ((perm[0] != 2 && perm[1] != 0
                && perm[2] != 1)
                || (perm[0] != 1 && perm[1] != 2 && perm[2] != 0))),
        "Only [2, 0, 1],[1, 2, 0] and [3, 2, 0, 1] permutation is supported");

    _input  = input;
    _output = output;
    _perm   = perm;

    const TensorShape output_shape = get_output_shape(input->info(), perm);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Create kernel
    std::set<std::string> build_opts;

    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.emplace("-DDEPTH_IN=" + support::cpp11::to_string(input->info()->dimension(2)));

    // Run [2, 0, 1] permute
    if(_perm[0] == 2 && _perm[1] == 0 && _perm[2] == 1)
    {
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("permute_201", build_opts));
    }
    // Run [1, 2, 0] permute
    else if(_perm[0] == 1 && _perm[1] == 2 && _perm[2] == 0)
    {
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("permute_120", build_opts));
    }
    // Run [3, 2, 0, 1] permute
    else if(_perm[0] == 3 && _perm[1] == 2 && _perm[2] == 0 && _perm[3] == 1)
    {
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("permute_3201", build_opts));
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported.");
    }

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    ICLKernel::configure(win);
}

void CLPermuteKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice_in = window.first_slice_window_4D();
    Window slice_out(slice_in);

    // Setup output slice
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_out.set(3, Window::Dimension(0, 0, 0));

    do
    {
        auto         collapsed_slice_in  = slice_in.collapse(ICLKernel::window(), 2);
        auto         collapsed_slice_out = slice_out.collapse(ICLKernel::window(), 2);
        unsigned int idx                 = 0;
        add_4D_tensor_argument(idx, _input, collapsed_slice_in);
        add_4D_tensor_argument(idx, _output, collapsed_slice_out);
        enqueue(queue, *this, collapsed_slice_in);
    }
    while(window.slide_window_slice_4D(slice_in) && window.slide_window_slice_4D(slice_out));
}
