/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace
{
constexpr unsigned int num_elems_read_per_iteration = 4;
constexpr unsigned int num_rows_read_per_iteration  = 4;

Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input0->data_type()) && (output->data_type() != DataType::S32));
    ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(2) != input1->dimension(1));

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output)
{
    const unsigned int border_x = ceil_to_multiple(input0->dimension(0), num_elems_read_per_iteration) - input0->dimension(0);
    const unsigned int border_y = ceil_to_multiple(input0->dimension(1), num_rows_read_per_iteration) - input0->dimension(1);

    Window win = calculate_max_window(*input0, Steps(num_elems_read_per_iteration));

    AccessWindowRectangle  input0_access(input0, 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal input1_access(input1, 0, num_elems_read_per_iteration);
    AccessWindowStatic     output_access(output, 0, 0, output->dimension(0) + border_x, output->dimension(1) + border_y);

    bool window_changed = update_window_and_padding(win, input0_access, input1_access, output_access);

    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info()));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Check if is a quantized operation
    bool is_quantized = is_data_type_quantized_asymmetric(_input0->info()->data_type());

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option_if(!is_quantized, "-DDATA_TYPE=" + get_cl_type_from_data_type(input0->info()->data_type()));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input0->info()->dimension(0)));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input0->info()->dimension(1)));

    std::string kernel_name = is_quantized ? std::string("gemm_mv_quantized") : std::string("gemm_mv");
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Add static arguments
    if(is_quantized)
    {
        const UniformQuantizationInfo iq0_info = _input0->info()->quantization_info().uniform();
        const UniformQuantizationInfo iq1_info = _input1->info()->quantization_info().uniform();

        unsigned int idx = num_arguments_per_3D_tensor() + num_arguments_per_2D_tensor() + num_arguments_per_1D_tensor();
        _kernel.setArg<int>(idx++, -iq0_info.offset);
        _kernel.setArg<int>(idx++, -iq1_info.offset);
    }

    // Configure kernel window
    _num_rows_read_per_iteration = num_rows_read_per_iteration;

    const unsigned int border_x = ceil_to_multiple(input0->info()->dimension(0), num_elems_read_per_iteration) - input0->info()->dimension(0);
    const unsigned int border_y = ceil_to_multiple(input0->info()->dimension(1), _num_rows_read_per_iteration) - input0->info()->dimension(1);

    _border_size = BorderSize(border_y, border_x);

    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);
}

Status CLGEMMMatrixVectorMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(), input1->clone().get(), output->clone().get()).first);

    return Status{};
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
        enqueue(queue, *this, slice_in, lws_hint());
    }
    while(window.slide_window_slice_3D(slice_in) && window.slide_window_slice_3D(slice_out));
}
} // namespace arm_compute
