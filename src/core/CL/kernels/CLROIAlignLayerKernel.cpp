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
#include "arm_compute/core/CL/kernels/CLROIAlignLayerKernel.h"

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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, rois, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, rois);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(0) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(compute_roi_align_shape(*input, *rois, pool_info), output->tensor_shape());
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = compute_roi_align_shape(*input, *rois, pool_info);
    auto_init_if_empty((*output), output_shape, 1, input->data_type());
    output->set_data_layout(input->data_layout());

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = 1;
    Window             win                               = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input_access(input, input->valid_region().start(0), num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLROIAlignLayerKernel::CLROIAlignLayerKernel()
    : _input(nullptr), _output(nullptr), _rois(nullptr), _pool_info(0, 0, 0.f)
{
}

void CLROIAlignLayerKernel::configure(const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, rois);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), rois->info(), output->info(), pool_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), rois->info(), output->info(), pool_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input     = input;
    _output    = output;
    _rois      = rois;
    _pool_info = pool_info;

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DDATA_SIZE=" + get_data_size_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DMAX_DIM_X=" + support::cpp11::to_string(_input->info()->dimension(get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH))));
    build_opts.add_option("-DMAX_DIM_Y=" + support::cpp11::to_string(_input->info()->dimension(get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT))));
    build_opts.add_option("-DMAX_DIM_Z=" + support::cpp11::to_string(_input->info()->dimension(get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL))));
    build_opts.add_option("-DPOOLED_DIM_X=" + support::cpp11::to_string(pool_info.pooled_width()));
    build_opts.add_option("-DPOOLED_DIM_Y=" + support::cpp11::to_string(pool_info.pooled_height()));
    build_opts.add_option("-DSPATIAL_SCALE=" + float_to_string_with_full_precision(pool_info.spatial_scale()));
    build_opts.add_option_if(input->info()->data_layout() == DataLayout::NHWC, "-DNHWC");
    build_opts.add_option_if(pool_info.sampling_ratio() > 0, "-DSAMPLING_RATIO=" + support::cpp11::to_string(pool_info.sampling_ratio()));

    // Create kernel
    std::string kernel_name = "roi_align_layer";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    ICLKernel::configure_internal(win_config.second);
}

Status CLROIAlignLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, rois, output, pool_info));
    return Status{};
}

void CLROIAlignLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice      = window.first_slice_window_3D();
    Window slice_rois = slice;
    // Parallelize spatially and across the fourth dimension of the output tensor (also across ROITensor)
    slice_rois.set_dimension_step(Window::DimX, _rois->info()->dimension(0));
    slice.set(get_data_layout_dimension_index(_input->info()->data_layout(), DataLayoutDimension::CHANNEL), window[3]);

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
