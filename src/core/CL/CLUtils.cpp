/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/StringSupport.h"

#include "src/core/CL/CLUtils.h"
#include "src/core/experimental/PostOp.h"

namespace arm_compute
{
cl::Image2D create_image2d_from_buffer(const cl::Context &ctx, const cl::Buffer &buffer, const TensorShape &shape2d, DataType data_type, size_t image_row_pitch)
{
    cl_channel_type cl_data_type;

    switch(data_type)
    {
        case DataType::F32:
            cl_data_type = CL_FLOAT;
            break;
        case DataType::F16:
            cl_data_type = CL_HALF_FLOAT;
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not support with OpenCL image2d");
    }

    cl_mem cl_image;
    cl_int err = CL_SUCCESS;

    const cl_image_format format = { CL_RGBA, cl_data_type };

    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    desc.mem_object      = buffer();
    desc.image_row_pitch = image_row_pitch;
    desc.image_width     = shape2d[0];
    desc.image_height    = shape2d[1];

    cl_image = clCreateImage(ctx(), CL_MEM_READ_ONLY, &format, &desc, nullptr, &err);

    ARM_COMPUTE_UNUSED(err);
    ARM_COMPUTE_ERROR_ON_MSG(err != CL_SUCCESS, "Error during the creation of CL image from buffer");

    return cl::Image2D(cl_image);
}

namespace experimental
{
PostOpCLKernelUtils::PostOpCLKernelUtils(const Config &supported_config)
    : _supported_config(supported_config)
{
    ARM_COMPUTE_ERROR_ON_MSG(supported_config.empty(), "Empty PostOp CL kernel support configuration is not allowed");
    for(auto it = _supported_config.begin(); it != _supported_config.end(); ++it)
    {
        auto post_op_sequence = it->first;
        auto post_op_slots    = std::get<1>(it->second);
        ARM_COMPUTE_ERROR_ON_MSG(post_op_sequence.size() != post_op_slots.size(), "The number of PostOps must be the same as that of the assigned slots");
    }
}

bool PostOpCLKernelUtils::are_post_op_shapes_compliant(const ITensorInfo *dst, const experimental::PostOpList<ITensorInfo *> &post_ops)
{
    for(const auto &op : post_ops.get_list())
    {
        for(const auto &tensor : op->arguments())
        {
            const TensorShape &out_shape = TensorShape::broadcast_shape(dst->tensor_shape(), (*tensor)->tensor_shape());
            // All post ops must be elementwise and must not alter the shape of the original dst tensor after broadcasting
            if(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0))
            {
                return false;
            }
            // NOTE: Kernel limitation: currently only the following broadcasting types are supported:
            //  1. Post op arg is scalar, broadcast in both X and Y
            //  2. Post op arg is of shape: Y=1, X=N, broadcast only in Y
            //  This means this case: Post op arg is of shape: Y=M, X=1, broadcast only in X, is NOT supported
            if(dst->dimension(0) > 1 && dst->dimension(1) > 1 && (*tensor)->dimension(0) == 1 && (*tensor)->dimension(1) > 1)
            {
                return false;
            }
        }
    }
    return true;
}

bool PostOpCLKernelUtils::is_post_op_sequence_supported(const PostOpList<ITensorInfo *> &post_ops) const
{
    if(post_ops.size() == 0)
    {
        return true; // Always support cases where no post op is specified
    }
    const auto post_op_sequence = get_post_op_sequence(post_ops);

    return _supported_config.find(post_op_sequence) != _supported_config.end();
}

void PostOpCLKernelUtils::set_post_ops_cl_build_options(CLBuildOptions &build_opts, const PostOpList<ITensorInfo *> &post_ops) const
{
    const auto post_op_sequence = get_post_op_sequence(post_ops);
    const auto slots            = std::get<1>(_supported_config.at(post_op_sequence));
    for(size_t post_op_id = 0; post_op_id < post_ops.size(); ++post_op_id)
    {
        const auto &post_op     = post_ops.get_list().at(post_op_id);
        const auto  slot_prefix = "-DP" + support::cpp11::to_string(slots[post_op_id]);
        if(post_op->type() == experimental::PostOpType::Activation)
        {
            const auto _post_op  = utils::cast::polymorphic_downcast<const experimental::PostOpAct<ITensorInfo *> *>(post_op.get());
            const auto act_type  = slot_prefix + "_ACTIVATION_TYPE=" + lower_string(string_from_activation_func(_post_op->_act_info.activation()));
            const auto act_a_val = slot_prefix + "_ACTIVATION_A_VAL=" + float_to_string_with_full_precision(_post_op->_act_info.a());
            const auto act_b_val = slot_prefix + "_ACTIVATION_B_VAL=" + float_to_string_with_full_precision(_post_op->_act_info.b());
            build_opts.add_option(act_type);
            build_opts.add_option(act_a_val);
            build_opts.add_option(act_b_val);
        }
        else if(post_op->type() == experimental::PostOpType::Eltwise_Add)
        {
            size_t     arg_id     = 1;
            const auto eltwise_op = slot_prefix + "_ELTWISE_OP=ADD" + "_X_POS_" + support::cpp11::to_string(post_op->prev_dst_pos());
            build_opts.add_option(eltwise_op);
            for(const auto &tensor : post_op->arguments())
            {
                const auto height = slot_prefix + "_ELTWISE_ARG" + support::cpp11::to_string(arg_id) + "_HEIGHT=" + support::cpp11::to_string((*tensor)->dimension(1));
                const auto width  = slot_prefix + "_ELTWISE_ARG" + support::cpp11::to_string(arg_id) + "_WIDTH=" + support::cpp11::to_string((*tensor)->dimension(0));
                build_opts.add_option(height);
                build_opts.add_option(width);
                ++arg_id;
            }
        }
        else if(post_op->type() == experimental::PostOpType::Eltwise_PRelu)
        {
            size_t     arg_id     = 1;
            const auto eltwise_op = slot_prefix + "_ELTWISE_OP=PRELU" + "_X_POS_" + support::cpp11::to_string(post_op->prev_dst_pos());
            build_opts.add_option(eltwise_op);
            for(const auto &tensor : post_op->arguments())
            {
                const auto height = slot_prefix + "_ELTWISE_ARG" + support::cpp11::to_string(arg_id) + "_HEIGHT=" + support::cpp11::to_string((*tensor)->dimension(1));
                const auto width  = slot_prefix + "_ELTWISE_ARG" + support::cpp11::to_string(arg_id) + "_WIDTH=" + support::cpp11::to_string((*tensor)->dimension(0));
                build_opts.add_option(height);
                build_opts.add_option(width);
                ++arg_id;
            }
        }
    }
}

void PostOpCLKernelUtils::set_post_ops_cl_kernel_name(std::string &kernel_name, const PostOpList<ITensorInfo *> &post_ops) const
{
    const auto post_op_sequence = get_post_op_sequence(post_ops);
    const auto postfix          = std::get<0>(_supported_config.at(post_op_sequence));
    kernel_name += postfix;
}
} // namespace experimental

} // namespace arm_compute