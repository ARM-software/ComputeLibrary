/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_EXPERIMENTAL_POSTOPUTILS
#define ARM_COMPUTE_EXPERIMENTAL_POSTOPUTILS

#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/core/experimental/PostOps.h"

#include "arm_compute/core/experimental/Types.h"
#include "support/Cast.h"

#include <vector>

/** (EXPERIMENTAL_POST_OPS) */
namespace arm_compute
{
namespace experimental
{
/** Transform a PostOpList of type FromTensorT to one of type ToTensorT */
template <typename FromTensorT, typename ToTensorT>
PostOpList<ToTensorT> transform_post_op_list_arguments(const PostOpList<FromTensorT> &post_ops, std::function<ToTensorT(FromTensorT)> transform_arg)
{
    PostOpList<ToTensorT> transformed_post_ops;
    for(const auto &post_op : post_ops.get_list())
    {
        switch(post_op->type())
        {
            case PostOpType::Activation:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const PostOpAct<FromTensorT> *>(post_op.get());
                transformed_post_ops.template push_back_op<PostOpAct<ToTensorT>>(_post_op->_act_info);
                break;
            }
            case PostOpType::Eltwise_Add:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const PostOpEltwiseAdd<FromTensorT> *>(post_op.get());
                transformed_post_ops.template push_back_op<PostOpEltwiseAdd<ToTensorT>>(transform_arg(_post_op->_addend), _post_op->_prev_dst_pos, _post_op->_policy);
                break;
            }
            case PostOpType::Eltwise_PRelu:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const PostOpEltwisePRelu<FromTensorT> *>(post_op.get());
                transformed_post_ops.template push_back_op<PostOpEltwisePRelu<ToTensorT>>(transform_arg(_post_op->_alpha_param), _post_op->_prev_dst_pos, _post_op->_policy);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported PostOpType");
            }
        }
    }
    return transformed_post_ops;
}

/** Get post op argument TensorType from post op argument index in a flattened, ordered post op argument list */
inline TensorType get_post_op_arg_type(size_t index)
{
    ARM_COMPUTE_ERROR_ON_MSG(static_cast<int>(index) > EXPERIMENTAL_ACL_POST_OP_ARG_LAST - EXPERIMENTAL_ACL_POST_OP_ARG_FIRST, "Post Op argument index is out of range");
    return static_cast<TensorType>(EXPERIMENTAL_ACL_POST_OP_ARG_FIRST + static_cast<int>(index));
}

/** Get a sequence of PostOp Types from PostOpList */
template <typename T>
PostOpTypeSequence get_post_op_sequence(const PostOpList<T> &post_ops)
{
    PostOpTypeSequence post_op_sequence;
    for(const auto &op : post_ops.get_list())
    {
        post_op_sequence.push_back(op->type());
    }
    return post_op_sequence;
}

} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_POSTOPUTILS
