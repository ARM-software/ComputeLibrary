/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
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

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() < 1 || input->num_dimensions() > 4,
                                    "Permutation upto 4-D input tensor is supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(perm.num_dimensions() < 1 || perm.num_dimensions() > 4,
                                    "Permutation vector size should be less than or equal to 4");
    for(const auto &p : perm)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(p >= perm.num_dimensions(), "Permutation vector has invalid values");
    }

    // Validate configured output
    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_permutation_output_shape(*input, perm);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}
} // namespace

void CLPermuteKernel::configure(const ICLTensor *input, ICLTensor *output, const PermutationVector &perm)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), perm));

    _input  = input;
    _output = output;
    _perm   = perm;

    const TensorShape output_shape = get_output_shape(input->info(), perm);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DDEPTH_IN=" + support::cpp11::to_string(input->info()->dimension(2)));
    // New positions of  width(W), height(H), channel(C) and batch(D) based on permutation vector
    build_opts.add_option("-DP1=" + support::cpp11::to_string((_perm.num_dimensions() >= 1) ? perm[0] : 0));
    build_opts.add_option("-DP2=" + support::cpp11::to_string((_perm.num_dimensions() >= 2) ? perm[1] : 1));
    build_opts.add_option("-DP3=" + support::cpp11::to_string((_perm.num_dimensions() >= 3) ? perm[2] : 2));
    build_opts.add_option("-DP4=" + support::cpp11::to_string((_perm.num_dimensions() >= 4) ? perm[3] : 3));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("permute", build_opts.options()));

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The CLPermute doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

Status CLPermuteKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, perm));

    return Status{};
}

void CLPermuteKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice_in = window.first_slice_window_4D().collapse(ICLKernel::window(), 2, 4);

    // Setup output slice
    Window slice_out(slice_in);
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
    slice_out.set(3, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice_in);
        add_4D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_in);
    }
    while(window.slide_window_slice_4D(slice_in) && window.slide_window_slice_4D(slice_out));
}
