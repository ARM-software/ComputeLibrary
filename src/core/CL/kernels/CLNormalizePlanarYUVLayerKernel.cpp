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
#include "arm_compute/core/CL/kernels/CLNormalizePlanarYUVLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, mean, std);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, std);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(mean->num_dimensions() > 1, "mean and std must be vectors");

    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(channel_idx) != mean->dimension(0));

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *mean, ITensorInfo *std)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, input->valid_region());

    if(input->data_layout() == DataLayout::NHWC)
    {
        AccessWindowHorizontal mean_access(mean, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal std_access(std, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, mean_access, std_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLNormalizePlanarYUVLayerKernel::CLNormalizePlanarYUVLayerKernel()
    : _input(nullptr), _output(nullptr), _mean(nullptr), _std(nullptr)
{
}

void CLNormalizePlanarYUVLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *std)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, mean, std);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), mean->info(), std->info()));

    _input  = input;
    _output = output;
    _mean   = mean;
    _std    = std;

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const unsigned int channel_idx                       = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);
    const DataType     dt                                = input->info()->data_type();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(dt)));
    build_opts.add_option(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));
    build_opts.add_option(("-DNUM_CHANNELS=" + support::cpp11::to_string(input->info()->dimension(channel_idx))));

    std::string kernel_name = "normalize_planar_yuv_layer_";
    if(is_data_type_quantized(dt))
    {
        const UniformQuantizationInfo qinfo = input->info()->quantization_info().uniform();
        build_opts.add_option(("-DOFFSET=" + support::cpp11::to_string(qinfo.offset)));
        build_opts.add_option(("-DSCALE=" + support::cpp11::to_string(qinfo.scale)));
        kernel_name += "q8_";
    }

    // Create kernel
    kernel_name += lower_string(string_from_data_layout(input->info()->data_layout()));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), mean->info(), std->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "normalize_planar_yuv_layer_";
    _config_id += lower_string(string_from_data_layout(input->info()->data_layout()));
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(dt));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
}

Status CLNormalizePlanarYUVLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, std));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), mean->clone().get(), std->clone().get()).first);

    return Status{};
}

void CLNormalizePlanarYUVLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    Window slice_in = collapsed.first_slice_window_1D();
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    add_1D_tensor_argument(idx, _mean, slice_in);
    add_1D_tensor_argument(idx, _std, slice_in);

    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
