/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClTransposeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
ClTransposeKernel::ClTransposeKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClTransposeKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output auto initialization if not yet initialized
    const TensorShape dst_shape = misc::shape_calculator::compute_transposed_shape(*src);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

    // Explicitly set the tensor shape to preserve dimensions
    dst->set_tensor_shape(dst_shape);

    ARM_COMPUTE_ERROR_THROW_ON(ClTransposeKernel::validate(src, dst));
    auto padding_info = get_padding_info({src, dst});

    unsigned int vec_size_x;
    unsigned int vec_size_y;

    // Set the optimal tile size for each data type without register spilling
    switch (src->element_size())
    {
        case 1:
            vec_size_x = adjust_vec_size(8, src->dimension(0));
            vec_size_y = adjust_vec_size(16, src->dimension(1));
            break;
        case 2:
            vec_size_x = adjust_vec_size(8, src->dimension(0));
            vec_size_y = adjust_vec_size(8, src->dimension(1));
            break;
        case 4:
            vec_size_x = adjust_vec_size(4, src->dimension(0));
            vec_size_y = adjust_vec_size(8, src->dimension(1));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
            break;
    }

    const int vec_size_x_leftovers = src->dimension(0) % vec_size_x;
    const int vec_size_y_leftovers = src->dimension(1) % vec_size_y;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN_BYTES=" + support::cpp11::to_string(src->element_size()));
    build_opts.add_option("-DVEC_SIZE_X=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER_X=" + support::cpp11::to_string(vec_size_x_leftovers));
    build_opts.add_option("-DVEC_SIZE_Y=" + support::cpp11::to_string(vec_size_y));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER_Y=" + support::cpp11::to_string(vec_size_y_leftovers));

    _kernel = create_kernel(compile_context, "transpose", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps(vec_size_x, vec_size_y));
    ICLKernel::configure_internal(win);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClTransposeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    // Validate configured dst
    if (dst->total_size() != 0)
    {
        const TensorInfo dst_info =
            src->clone()->set_tensor_shape(misc::shape_calculator::compute_transposed_shape(*src));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &dst_info);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}

void ClTransposeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Collapse dimensions higher than width and height into the batch dimension
    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    } while (collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
