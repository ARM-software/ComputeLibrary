/*
 * Copyright (c) 2021 Arm Limited.
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
#include "PostOps.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "src/core/experimental/PostOp.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> post_ops(const SimpleTensor<T> &a, experimental::PostOpList<SimpleTensor<T>> post_ops)
{
    // Create reference
    SimpleTensor<T> dst{ a };

    for(auto &post_op : post_ops.get_list())
    {
        switch(post_op->type())
        {
            case experimental::PostOpType::Activation:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpAct<SimpleTensor<T>> *>(post_op.get());
                dst                 = reference::activation_layer(dst, _post_op->_act_info);
                break;
            }
            case experimental::PostOpType::Eltwise_Add:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpEltwiseAdd<SimpleTensor<T>> *>(post_op.get());
                dst                 = reference::arithmetic_operation(ArithmeticOperation::ADD, dst, _post_op->_addend, dst, _post_op->_policy);
                break;
            }
            case experimental::PostOpType::Eltwise_PRelu:
            {
                const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpEltwisePRelu<SimpleTensor<T>> *>(post_op.get());

                // If previous main operation output is the the first pRelu argument, then pass it as src1 parameter of the arithmetic operation
                if(_post_op->_prev_dst_pos == 0)
                {
                    dst = reference::arithmetic_operation(ArithmeticOperation::PRELU, dst, _post_op->_alpha_param, dst, _post_op->_policy);
                }
                // If previous main operation output is the the second pRelu argument, then pass it as src2 parameter of the arithmetic operation
                else if(_post_op->_prev_dst_pos == 1)
                {
                    dst = reference::arithmetic_operation(ArithmeticOperation::PRELU, _post_op->_alpha_param, dst, dst, _post_op->_policy);
                }
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported PostOpType");
            }
        }
    }
    return dst;
}

template SimpleTensor<float> post_ops(const SimpleTensor<float> &a, experimental::PostOpList<SimpleTensor<float>> post_ops);
template SimpleTensor<half> post_ops(const SimpleTensor<half> &a, experimental::PostOpList<SimpleTensor<half>> post_ops);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute