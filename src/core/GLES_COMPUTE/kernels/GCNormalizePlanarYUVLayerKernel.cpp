/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizePlanarYUVLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/StringSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW);

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
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *mean, ITensorInfo *std)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    const unsigned int num_elems_processed_per_iteration = 4;

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    const int              mean_padding = ceil_to_multiple(mean->dimension(0), num_elems_processed_per_iteration) - mean->dimension(0);
    const int              std_padding  = ceil_to_multiple(std->dimension(0), num_elems_processed_per_iteration) - std->dimension(0);
    AccessWindowStatic     mean_access(mean, 0, 0, mean->dimension(0) + mean_padding, mean->dimension(1));
    AccessWindowStatic     std_access(std, 0, 0, std->dimension(0) + std_padding, std->dimension(1));

    const bool window_changed = update_window_and_padding(win, input_access, output_access, mean_access, std_access);
    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

GCNormalizePlanarYUVLayerKernel::GCNormalizePlanarYUVLayerKernel()
    : _input(nullptr), _output(nullptr), _mean(nullptr), _std(nullptr)
{
}

void GCNormalizePlanarYUVLayerKernel::configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *mean, const IGCTensor *std)
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

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("normalize_planar_yuv_layer", build_opts));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), mean->info(), std->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    IGCKernel::configure(std::get<1>(win_config));
}

Status GCNormalizePlanarYUVLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, std));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), mean->clone().get(), std->clone().get())));
    return Status{};
}

void GCNormalizePlanarYUVLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    _output->set_needs_shifting(true);

    Window slice = window.first_slice_window_3D();

    Window slice_in;
    //slice_in.use_tensor_dimensions(_mean->info()->tensor_shape());
    slice_in = window.first_slice_window_1D();
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    add_1D_tensor_argument(idx, _mean, 3, slice_in);
    add_1D_tensor_argument(idx, _std, 4, slice_in);

    slice_in = window.first_slice_window_3D();

    slice.shift(Window::DimX, -(_output->info()->padding()).left);

    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_in));
}
