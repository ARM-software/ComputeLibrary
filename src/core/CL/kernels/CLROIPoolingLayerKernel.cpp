/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLROIPoolingLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <cfloat>
#include <cmath>
#include <string>

namespace arm_compute
{
CLROIPoolingLayerKernel::CLROIPoolingLayerKernel()
    : _input(nullptr), _rois(nullptr), _output(nullptr), _pool_info(0, 0, 0.f)
{
}

Status CLROIPoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *rois, const ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, rois, output);

    //Validate arguments
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, rois, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rois, 1, DataType::U16);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(0) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON((output->dimension(0) != pool_info.pooled_width()) || (output->dimension(1) != pool_info.pooled_height()));
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) != output->dimension(2));
        ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(1) != output->dimension(3));
    }

    return Status{};
}

void CLROIPoolingLayerKernel::configure(const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, rois, output, pool_info);
}

void CLROIPoolingLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *rois, const ICLTensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_THROW_ON(CLROIPoolingLayerKernel::validate(input->info(), rois->info(), output->info(), pool_info));

    auto padding_info = get_padding_info({ input, rois, output });

    // Output auto initialization if not yet initialized
    TensorShape output_shape(pool_info.pooled_width(), pool_info.pooled_height(), input->info()->dimension(2), rois->info()->dimension(1));
    auto_init_if_empty(*(output->info()), output_shape, 1, input->info()->data_type(), output->info()->quantization_info());

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    const DataType data_type = input->info()->data_type();
    const bool     is_qasymm = is_data_type_quantized_asymmetric(data_type);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DDATA_SIZE=" + get_data_size_from_data_type(data_type));
    build_opts.add_option("-DMAX_DIM_X=" + support::cpp11::to_string(_input->info()->dimension(Window::DimX)));
    build_opts.add_option("-DMAX_DIM_Y=" + support::cpp11::to_string(_input->info()->dimension(Window::DimY)));
    build_opts.add_option("-DMAX_DIM_Z=" + support::cpp11::to_string(_input->info()->dimension(Window::DimZ)));
    build_opts.add_option("-DPOOLED_DIM_X=" + support::cpp11::to_string(pool_info.pooled_width()));
    build_opts.add_option("-DPOOLED_DIM_Y=" + support::cpp11::to_string(pool_info.pooled_height()));
    build_opts.add_option("-DSPATIAL_SCALE=" + support::cpp11::to_string(pool_info.spatial_scale()));

    if(is_qasymm)
    {
        // Determine quantization info scale, offset
        UniformQuantizationInfo uqinfo = UniformQuantizationInfo();
        uqinfo                         = compute_requantization_scale_offset(_input->info()->quantization_info().uniform(), _output->info()->quantization_info().uniform());
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(uqinfo.offset));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(uqinfo.scale));

        // Specify minimum possible value of datatype
        build_opts.add_option("-DMIN_VALUE=" + support::cpp11::to_string(0));
    }
    else
    {
        // Specify min value of F32 datatype
        build_opts.add_option("-DMIN_VALUE=" + support::cpp11::to_string(-FLT_MAX));
    }

    Window win = calculate_max_window(*(output->info()), Steps());
    ICLKernel::configure_internal(win);

    // Create kernel
    std::string kernel_name = "roi_pooling_layer";
    _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
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

    enqueue(queue, *this, slice, lws_hint());
}
} // namespace arm_compute
