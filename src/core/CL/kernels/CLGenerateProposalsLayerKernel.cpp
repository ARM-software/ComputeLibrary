/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGenerateProposalsLayerKernel.h"

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

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(anchors);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->dimension(0) != info.values_per_roi());
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(anchors, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->num_dimensions() > 2);
    if(all_anchors->total_size() > 0)
    {
        size_t feature_height = info.feat_height();
        size_t feature_width  = info.feat_width();
        size_t num_anchors    = anchors->dimension(1);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(all_anchors, anchors);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(0) != info.values_per_roi());
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(1) != feature_height * feature_width * num_anchors);
    }
    return Status{};
}
} // namespace

CLComputeAllAnchorsKernel::CLComputeAllAnchorsKernel()
    : _anchors(nullptr), _all_anchors(nullptr)
{
}

void CLComputeAllAnchorsKernel::configure(const ICLTensor *anchors, ICLTensor *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(anchors->info(), all_anchors->info(), info));

    // Metadata
    const size_t   num_anchors = anchors->info()->dimension(1);
    const DataType data_type   = anchors->info()->data_type();
    const float    width       = info.feat_width();
    const float    height      = info.feat_height();

    // Initialize the output if empty
    const TensorShape output_shape(info.values_per_roi(), width * height * num_anchors);
    auto_init_if_empty(*all_anchors->info(), output_shape, 1, data_type);

    // Set instance variables
    _anchors     = anchors;
    _all_anchors = all_anchors;

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DWIDTH=" + float_to_string_with_full_precision(width));
    build_opts.add_option("-DHEIGHT=" + float_to_string_with_full_precision(height));
    build_opts.add_option("-DSTRIDE=" + float_to_string_with_full_precision(1.f / info.spatial_scale()));
    build_opts.add_option("-DNUM_ANCHORS=" + support::cpp11::to_string(num_anchors));
    build_opts.add_option("-DNUM_ROI_FIELDS=" + support::cpp11::to_string(info.values_per_roi()));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("generate_proposals_compute_all_anchors", build_opts.options()));

    // The tensor all_anchors can be interpreted as an array of structs (each structs has values_per_roi fields).
    // This means we don't need to pad on the X dimension, as we know in advance how many fields
    // compose the struct.
    Window win = calculate_max_window(*all_anchors->info(), Steps(info.values_per_roi()));
    ICLKernel::configure_internal(win);
}

Status CLComputeAllAnchorsKernel::validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(anchors, all_anchors, info));
    return Status{};
}

void CLComputeAllAnchorsKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Collapse everything on the first dimension
    Window collapsed = window.collapse(ICLKernel::window(), Window::DimX);

    // Set arguments
    unsigned int idx = 0;
    add_1D_tensor_argument(idx, _anchors, collapsed);
    add_1D_tensor_argument(idx, _all_anchors, collapsed);

    // Note that we don't need to loop over the slices, as we are launching exactly
    // as many threads as all the anchors generated
    enqueue(queue, *this, collapsed);
}
} // namespace arm_compute
