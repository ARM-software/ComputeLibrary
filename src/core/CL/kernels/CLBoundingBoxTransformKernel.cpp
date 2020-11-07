/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLBoundingBoxTransformKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(boxes);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(boxes, DataType::QASYMM16, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(deltas, DataType::QASYMM8, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[1] != boxes->tensor_shape()[1]);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[0] % 4 != 0);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->tensor_shape()[0] != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->num_dimensions() > 2);

    const bool is_qasymm16 = boxes->data_type() == DataType::QASYMM16;
    if(is_qasymm16)
    {
        const UniformQuantizationInfo boxes_qinfo = boxes->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(boxes_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(boxes_qinfo.offset != 0);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(deltas, DataType::QASYMM8);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(boxes, deltas);
    }

    if(pred_boxes->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(pred_boxes->tensor_shape(), deltas->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(pred_boxes, boxes);
        ARM_COMPUTE_RETURN_ERROR_ON(pred_boxes->num_dimensions() > 2);
        if(is_qasymm16)
        {
            const UniformQuantizationInfo pred_boxes_qinfo = pred_boxes->quantization_info().uniform();
            ARM_COMPUTE_RETURN_ERROR_ON(pred_boxes_qinfo.scale != 0.125f);
            ARM_COMPUTE_RETURN_ERROR_ON(pred_boxes_qinfo.offset != 0);
        }
    }
    ARM_COMPUTE_RETURN_ERROR_ON(info.scale() <= 0);

    return Status{};
}
} // namespace

CLBoundingBoxTransformKernel::CLBoundingBoxTransformKernel()
    : _boxes(nullptr), _pred_boxes(nullptr), _deltas(nullptr)
{
}

void CLBoundingBoxTransformKernel::configure(const ICLTensor *boxes, ICLTensor *pred_boxes, const ICLTensor *deltas, const BoundingBoxTransformInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), boxes, pred_boxes, deltas, info);
}

void CLBoundingBoxTransformKernel::configure(const CLCompileContext &compile_context, const ICLTensor *boxes, ICLTensor *pred_boxes, const ICLTensor *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    auto_init_if_empty(*pred_boxes->info(), deltas->info()->clone()->set_data_type(boxes->info()->data_type()).set_quantization_info(boxes->info()->quantization_info()));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(boxes->info(), pred_boxes->info(), deltas->info(), info));

    // Set instance variables
    _boxes      = boxes;
    _pred_boxes = pred_boxes;
    _deltas     = deltas;

    // Get image height and widht (rescaled)
    const int img_h = floor(info.img_height() / info.scale() + 0.5f);
    const int img_w = floor(info.img_width() / info.scale() + 0.5f);

    const bool is_quantized = is_data_type_quantized(boxes->info()->data_type());

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(boxes->info()->data_type()));
    build_opts.add_option("-DWEIGHT_X=" + float_to_string_with_full_precision(info.weights()[0]));
    build_opts.add_option("-DWEIGHT_Y=" + float_to_string_with_full_precision(info.weights()[1]));
    build_opts.add_option("-DWEIGHT_W=" + float_to_string_with_full_precision(info.weights()[2]));
    build_opts.add_option("-DWEIGHT_H=" + float_to_string_with_full_precision(info.weights()[3]));
    build_opts.add_option("-DBBOX_XFORM_CLIP=" + float_to_string_with_full_precision(info.bbox_xform_clip()));
    build_opts.add_option("-DIMG_WIDTH=" + support::cpp11::to_string(img_w));
    build_opts.add_option("-DIMG_HEIGHT=" + support::cpp11::to_string(img_h));
    build_opts.add_option("-DBOX_FIELDS=" + support::cpp11::to_string(4));
    build_opts.add_option("-DSCALE_BEFORE=" + float_to_string_with_full_precision(info.scale()));
    build_opts.add_option_if(info.apply_scale(), "-DSCALE_AFTER=" + float_to_string_with_full_precision(info.scale()));
    build_opts.add_option_if(info.correct_transform_coords(), "-DOFFSET=1");

    if(is_quantized)
    {
        build_opts.add_option("-DDATA_TYPE_DELTAS=" + get_cl_type_from_data_type(deltas->info()->data_type()));
        const UniformQuantizationInfo boxes_qinfo      = boxes->info()->quantization_info().uniform();
        const UniformQuantizationInfo deltas_qinfo     = deltas->info()->quantization_info().uniform();
        const UniformQuantizationInfo pred_boxes_qinfo = pred_boxes->info()->quantization_info().uniform();
        build_opts.add_option("-DOFFSET_BOXES=" + float_to_string_with_full_precision(boxes_qinfo.offset));
        build_opts.add_option("-DSCALE_BOXES=" + float_to_string_with_full_precision(boxes_qinfo.scale));
        build_opts.add_option("-DOFFSET_DELTAS=" + float_to_string_with_full_precision(deltas_qinfo.offset));
        build_opts.add_option("-DSCALE_DELTAS=" + float_to_string_with_full_precision(deltas_qinfo.scale));
        build_opts.add_option("-DOFFSET_PRED_BOXES=" + float_to_string_with_full_precision(pred_boxes_qinfo.offset));
        build_opts.add_option("-DSCALE_PRED_BOXES=" + float_to_string_with_full_precision(pred_boxes_qinfo.scale));
    }

    // Create kernel
    const std::string kernel_name = (is_quantized) ? "bounding_box_transform_quantized" : "bounding_box_transform";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts.options());

    // Since the number of columns is a multiple of 4 by definition, we don't need to pad the tensor
    const unsigned int num_elems_processed_per_iteration = 4;
    Window             win                               = calculate_max_window(*deltas->info(), Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);
}

Status CLBoundingBoxTransformKernel::validate(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(boxes, pred_boxes, deltas, info));
    return Status{};
}

void CLBoundingBoxTransformKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    // Set arguments
    unsigned int idx = 0;
    add_1D_tensor_argument(idx, _boxes, slice);
    add_2D_tensor_argument(idx, _pred_boxes, slice);
    add_2D_tensor_argument(idx, _deltas, slice);

    // Note that we don't need to loop over the slices, as we are sure that we are dealing with all 2D tensors
    enqueue(queue, *this, slice, lws_hint());
}
} // namespace arm_compute
