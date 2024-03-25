/*
 * Copyright (c) 2024 Arm Limited.
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
#include "src/gpu/cl/kernels/ClScatterKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
ClScatterKernel::ClScatterKernel()
{
}

Status ClScatterKernel::validate(const ITensorInfo *updates,
                                 const ITensorInfo *indices,
                                 const ITensorInfo *dst,
                                 const ScatterInfo &info)
{
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(updates, dst);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(indices, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(dst, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->num_dimensions() > 1, "Only 1D output tensors are currently supported.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(indices->num_dimensions() > 2, "Only 2D indices tensors are currently supported.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(updates->num_dimensions() > 1, "Only 1D update tensors are currently supported.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        indices->tensor_shape().y() != updates->tensor_shape()[updates->num_dimensions() - 1],
        "Height of indices tensor should match size of highest dimension in updates tensor.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(updates->num_dimensions() > dst->num_dimensions(),
                                    "Update tensor cannot have more dims than output tensor.");
    ARM_COMPUTE_UNUSED(info);

    return Status{};
}
void ClScatterKernel::configure(const ClCompileContext &compile_context,
                                const ITensorInfo      *updates,
                                const ITensorInfo      *indices,
                                ITensorInfo            *dst,
                                const ScatterInfo      &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(updates, dst, indices);
    ARM_COMPUTE_LOG_PARAMS(updates, indices, dst, info);

    // Configure kernel window
    const auto indices_shape = indices->tensor_shape();
    Window     win           = calculate_max_window(
                      *indices, Steps(indices_shape.x(), indices_shape.y())); // Ensures single thread for deterministic output.

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option("-DINDICES_DIMS=" + support::cpp11::to_string(indices->num_dimensions()));
    build_opts.add_option("-DINDICES_SHAPE_Y=" + support::cpp11::to_string(indices_shape.y()));
    build_opts.add_option("-DOUT_SHAPE_X=" + support::cpp11::to_string(dst->tensor_shape().x()));

    switch (info.func)
    {
        case ScatterFunction::Update:
            build_opts.add_option("-DSCATTER_FUNCTION=UPDATE_OP");
            break;
        case ScatterFunction::Add:
            build_opts.add_option("-DSCATTER_FUNCTION=ADD_OP");
            break;
        case ScatterFunction::Sub:
            build_opts.add_option("-DSCATTER_FUNCTION=SUB_OP");
            break;
        case ScatterFunction::Max:
            build_opts.add_option("-DSCATTER_FUNCTION=MAX_OP");
            break;
        case ScatterFunction::Min:
            build_opts.add_option("-DSCATTER_FUNCTION=MIN_OP");
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Create kernel
    std::string kernel_name("scatter1D");
    ICLKernel::configure_internal(win);
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());
    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(updates->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
}

void ClScatterKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    unsigned int idx = 0;

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    const auto updates =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto indices =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    add_4D_tensor_argument(idx, updates, window_collapsed);
    add_4D_tensor_argument(idx, indices, window_collapsed);
    add_4D_tensor_argument(idx, dst, window_collapsed);

    enqueue(queue, *this, window, lws_hint());
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
