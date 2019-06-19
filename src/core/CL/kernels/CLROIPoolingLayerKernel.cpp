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
#include "arm_compute/core/CL/kernels/CLROIPoolingLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include <cmath>
#include <set>
#include <string>

namespace arm_compute
{
namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto initialization if not yet initialized
    TensorShape output_shape(pool_info.pooled_width(), pool_info.pooled_height(), input->dimension(2), rois->dimension(1));
    auto_init_if_empty((*output), output_shape, 1, input->data_type());

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;
    Window                 win                               = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input_access(input, input->valid_region().start(0), num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLROIPoolingLayerKernel::CLROIPoolingLayerKernel()
    : _input(nullptr), _rois(nullptr), _output(nullptr), _pool_info(0, 0, 0.f)
{
}

void CLROIPoolingLayerKernel::configure(const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, rois, output);

    //Validate arguments
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), rois->info(), output->info());
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rois, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON(rois->info()->dimension(0) != 5);
    ARM_COMPUTE_ERROR_ON(rois->info()->num_dimensions() > 2);
    ARM_COMPUTE_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));

    if(output->info()->total_size() != 0)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pool_info.pooled_width()) || (output->info()->dimension(1) != pool_info.pooled_height()));
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != output->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(rois->info()->dimension(1) != output->info()->dimension(3));
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), rois->info(), output->info(), pool_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DDATA_SIZE=" + get_data_size_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DMAX_DIM_X=" + support::cpp11::to_string(_input->info()->dimension(Window::DimX))));
    build_opts.emplace(("-DMAX_DIM_Y=" + support::cpp11::to_string(_input->info()->dimension(Window::DimY))));
    build_opts.emplace(("-DMAX_DIM_Z=" + support::cpp11::to_string(_input->info()->dimension(Window::DimZ))));
    build_opts.emplace(("-DPOOLED_DIM_X=" + support::cpp11::to_string(pool_info.pooled_width())));
    build_opts.emplace(("-DPOOLED_DIM_Y=" + support::cpp11::to_string(pool_info.pooled_height())));
    build_opts.emplace(("-DSPATIAL_SCALE=" + support::cpp11::to_string(pool_info.spatial_scale())));

    // Create kernel
    std::string kernel_name = "roi_pooling_layer";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor() + num_arguments_per_1D_array();
    add_argument<cl_uint>(idx, _input->info()->strides_in_bytes()[3]);
    add_argument<cl_uint>(idx, _output->info()->strides_in_bytes()[3]);

    ICLKernel::configure_internal(win_config.second);
}

void CLROIPoolingLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice      = window.first_slice_window_3D();
    Window slice_rois = slice;
    // Parallelize spatially and across the fourth dimension of the output tensor (also across ROITensor)
    slice_rois.set_dimension_step(Window::DimX, _rois->info()->dimension(0));
    slice.set(Window::DimZ, window[3]);

    // Set arguments
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, slice);
    add_2D_tensor_argument(idx, _rois, slice_rois);
    add_3D_tensor_argument(idx, _output, slice);
    add_argument<cl_uint>(idx, _input->info()->strides_in_bytes()[3]);
    add_argument<cl_uint>(idx, _output->info()->strides_in_bytes()[3]);

    enqueue(queue, *this, slice);
}
} // namespace arm_compute
