/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixVectorMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

using namespace arm_compute;

CLGEMMMatrixVectorMultiplyKernel::CLGEMMMatrixVectorMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _num_rows_read_per_iteration(0), _border_size(0)
{
}
BorderSize CLGEMMMatrixVectorMultiplyKernel::border_size() const
{
    return _border_size;
}

void CLGEMMMatrixVectorMultiplyKernel::configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1, output);
    ARM_COMPUTE_ERROR_ON(input0->info()->dimension(2) != input1->info()->dimension(1));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Create kernel
    std::set<std::string> build_opts;

    build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input0->info()->data_type()));
    build_opts.emplace("-DSRC_WIDTH=" + support::cpp11::to_string(input0->info()->dimension(0)));
    build_opts.emplace("-DSRC_HEIGHT=" + support::cpp11::to_string(input0->info()->dimension(1)));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemm_mv", build_opts));

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning for the MobileNets tensor shapes.
    const GPUTarget gpu_target = get_arch_from_target(get_target());
    if(gpu_target == GPUTarget::BIFROST)
    {
        _lws_hint = cl::NDRange(1, 1, 1);
    }

    // Configure kernel window
    const unsigned int num_elems_read_per_iteration = 4;

    _num_rows_read_per_iteration = 4;

    const unsigned int border_x = ceil_to_multiple(input0->info()->dimension(0), num_elems_read_per_iteration) - input0->info()->dimension(0);
    const unsigned int border_y = ceil_to_multiple(input0->info()->dimension(1), _num_rows_read_per_iteration) - input0->info()->dimension(1);

    _border_size = BorderSize(border_y, border_x);

    Window win = calculate_max_window(*input0->info(), Steps(num_elems_read_per_iteration));

    AccessWindowRectangle  input0_access(input0->info(), 0, 0, num_elems_read_per_iteration, _num_rows_read_per_iteration);
    AccessWindowHorizontal input1_access(input1->info(), 0, num_elems_read_per_iteration);
    AccessWindowStatic     output_access(_output->info(), 0, 0, _output->info()->dimension(0) + border_x, _output->info()->dimension(1) + border_y);

    update_window_and_padding(win, input0_access, input1_access, output_access);

    _output->info()->set_valid_region(ValidRegion(Coordinates(), _output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLGEMMMatrixVectorMultiplyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice_in  = window.first_slice_window_3D();
    Window slice_in2 = window.first_slice_window_3D();
    Window slice_out = window.first_slice_window_3D();

    // Setup input0 slice
    slice_in.set(Window::DimX, Window::Dimension(0, _input0->info()->dimension(0), _input0->info()->dimension(0)));
    slice_in.set(Window::DimY, Window::Dimension(0, _input0->info()->dimension(1) + border_size().bottom, _num_rows_read_per_iteration));
    slice_in.set(Window::DimZ, Window::Dimension(0, _input0->info()->dimension(2), 1));

    // Setup input1 and output slice. Their dimensions are increased in the cl kernel.
    slice_in2.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in2.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in2.set(Window::DimZ, Window::Dimension(0, 0, 0));

    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    unsigned int idx_1 = num_arguments_per_3D_tensor();

    add_2D_tensor_argument(idx_1, _input1, slice_in2);

    do
    {
        unsigned int idx_0 = 0;
        unsigned int idx_2 = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor();
        add_3D_tensor_argument(idx_0, _input0, slice_in);
        add_1D_tensor_argument(idx_2, _output, slice_out);
        enqueue(queue, *this, slice_in, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice_in) && window.slide_window_slice_3D(slice_out));
}
