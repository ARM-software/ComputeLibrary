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
#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include <cstdint>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

namespace
{
constexpr int max_index_length = 5;
} // namespace

ClScatterKernel::ClScatterKernel()
{
}

Status ClScatterKernel::validate(const ITensorInfo *updates,
                                 const ITensorInfo *indices,
                                 const ITensorInfo *dst,
                                 const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    const TensorShape &ind_shape = indices->tensor_shape();
    const TensorShape &upt_shape = updates->tensor_shape();
    const TensorShape &dst_shape = dst->tensor_shape();

    const int32_t upt_dims = upt_shape.num_dimensions();
    const int32_t dst_dims = dst_shape.num_dimensions();
    const int32_t ind_dims = ind_shape.num_dimensions();
    const int32_t data_dim = upt_dims - (ind_dims - 1); // Number of batch dims is the number of indices dims - 1

    const int32_t index_len = ind_shape[0];
    bool          unsupported_padding_config =
        (dst_dims == index_len) && index_len > 1 && (dst->has_padding() || updates->has_padding());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(unsupported_padding_config, "Padding is not supported with these shapes.");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(updates, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(indices, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(dst, DataType::F32, DataType::F16, DataType::S32, DataType::S16,
                                                 DataType::S8, DataType::U32, DataType::U16, DataType::U8);

    // Check data dims in update tensor and output tensor are equal
    for (int32_t i = 0; i < data_dim; i++)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(upt_shape[i] != dst_shape[i],
                                        "Data dims should be same size in both updates and ouput tensor.");
    }

    // Check if batch dims in indices and updates tensor are equal.
    for (int32_t i = 0; i < ind_dims - 1; i++)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(upt_shape[data_dim + i] != ind_shape[i + 1],
                                        "Batch dimensions should be the same in updates and indices tensor.");
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(ind_shape[1] != upt_shape[data_dim],
                                    "Height of indices tensor should match size of highest dimension in updates tensor "
                                    "(Excluding batch dimension)");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        data_dim >= dst_dims, "Update tensor cannot have more dims than output tensor. (Excluding batch dimensions)");
    ARM_COMPUTE_RETURN_ERROR_ON(index_len != dst_dims - data_dim);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((ind_dims < 2), "Shape of Indices tensor must be at least 2D");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(index_len > max_index_length, "Maximum supported index length is 5!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(index_len > dst_dims && dst_dims != 1,
                                    "Index length should be smaller than or equal to number of output dims");

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

    const TensorShape &dst_shape = dst->tensor_shape();
    const int          index_len = indices->dimension(0);

    // Check for single element data block
    const bool is_scalar_block = (dst->num_dimensions() == static_cast<uint32_t>(index_len));

    const int n0         = adjust_vec_size(16 / updates->element_size(), is_scalar_block ? 1 : updates->dimension(0));
    const int partial_n0 = updates->dimension(0) % n0;

    // The GWS will be 2D [x, y]
    //  x-dimension refers to the x coordinate of the dst tensor
    //  y-dimension refers to the collapsed y-coordinate of the data part of the dst tensor
    Window win;

    if (!is_scalar_block)
    {
        win = calculate_max_window(dst_shape, Steps(n0));

        // Collapse the dimensions corresponding to indices in the execution window
        for (int i = 0; i < index_len; ++i)
        {
            win.set(dst->num_dimensions() - (i + 1), Window::Dimension(0, 1, 1));
        }

        win = win.collapse(win, 1);
    }

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option_if(is_data_type_float(dst->data_type()), "-DIS_FLOAT");

    const int   num_dims      = dst->num_dimensions();
    TensorShape ind_collapsed = indices->tensor_shape().collapsed_from(1);
    build_opts.add_option("-DNUM_INDICES=" + support::cpp11::to_string(ind_collapsed[1]));
    build_opts.add_option("-DINDEX_LENGTH=" + support::cpp11::to_string(index_len));

    // We provide 5 variables to use in a constant array
    for (int i = 1; i <= max_index_length; i++)
    {
        build_opts.add_option("-DOUT_SHAPE_N_MINUS_" + support::cpp11::to_string(i) + "=" +
                              support::cpp11::to_string(dst_shape[std::max(num_dims - i, 0)]));
    }

    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_n0));

    switch (info.func)
    {
        case ScatterFunction::Update:
            build_opts.add_option("-DSCATTER_FUNCTION=UPDATE_OP");
            build_opts.add_option("-DSKIP_OUTPUT_READ");
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
    std::string kernel_name = "scatter_mp1d_2d_mpnd";
    build_opts.add_option("-D" + upper_string(kernel_name));

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
    const auto updates =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto indices =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    const ITensorInfo *dst_info  = dst->info();
    const ITensorInfo *upd_info  = updates->info();
    const int          num_dims  = dst_info->num_dimensions();
    const int          ind_dims  = indices->info()->num_dimensions();
    const int          index_len = indices->info()->dimension(0);

    bool unsupported_padding_config =
        num_dims == index_len && index_len > 1 && (dst_info->has_padding() || upd_info->has_padding());
    if (unsupported_padding_config)
    {
        ARM_COMPUTE_ERROR("Unsupported Configuration! Padding not supported with these shapes.");
    }

    // calculate m-dimensional data block strides in updates and destination tensors
    const int upt_block_stride =
        updates->info()->strides_in_bytes()[updates->info()->num_dimensions() - (ind_dims - 1)];

    const int out_block_stride = dst_info->strides_in_bytes()[num_dims - index_len];

    unsigned int idx = 0;

    add_2D_tensor_argument(idx, updates, window);
    add_2D_tensor_argument(idx, indices, window);
    add_2D_tensor_argument(idx, dst, window);

    _kernel.setArg<cl_int>(idx++, upt_block_stride);
    _kernel.setArg<cl_int>(idx++, out_block_stride);

    enqueue(queue, *this, window, lws_hint());
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
