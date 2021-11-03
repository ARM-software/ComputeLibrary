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
#ifndef ARM_COMPUTE_EXPERIMENTAL_IPOSTOP
#define ARM_COMPUTE_EXPERIMENTAL_IPOSTOP

#include <memory>
#include <numeric>
#include <vector>

namespace arm_compute
{
namespace experimental
{
/** Type of Post Op */
enum class PostOpType
{
    Activation,
    Eltwise_Add,
    Eltwise_PRelu
};
/** An ordered sequence of type of Post Ops */
using PostOpTypeSequence = std::vector<PostOpType>;
/** An elementwise n-ary operation that can be appended to and fused with (at kernel-level) other operators
 *  It contains:
 *      1. The attributes of the original operator.
 *      2. Any additional tensor argument.
 *      3. The position of the previous op's dst tensor in its argument list ( @ref prev_dst_pos )
 *
 *  For example, a series of chained ops:
 *
 *          div(src1, relu(conv(src0, weights, bias, conv_info), act_info), div_info)
 *
 *      translates to
 *
 *          dst = conv(src0, weights, bias, conv_info)  // main op
 *          dst = relu(dst, act_info)                   // previous dst is placed in the first (and only) argument
 *          dst = div(src1, dst, div_info)              // previous dst is placed in the second argument
 *
 *      which in turn translates to:
 *
 *          main op: conv(src0, weights, bias, conv_info)
 *          post op1: relu(act_info, prev_dst_pos = 0)
 *          post op2: div(div_info, src1, prev_dst_pos = 1)
 *
 *  @note: On Broadcasting
 *      For n-ary post ops, the tensor arguments must not "widen" the dst tensor of the main op
 *      For example, for a dst of shape [14, 1, 34]:
 *          * post_op_arg1 = [1, 1, 34] is allowed: broadcast in dim 0
 *          * post_op_arg1 = [14, 1, 34] is allowed: no broadcast
 *          * post_op_arg1 = [1, 1, 34] is allowed: broadcast in dims 0 and 1
 *          * post_op_arg1 = [14, 15, 34] is NOT allowed: broadcast widens the dst tensor
 *
 * @note: On Data layout
 *      All post ops are data layout agnostic. This means post ops do not have an inherent idea of "width", "height" and so on.
 *      Should we want to perform a post op with 2 tensors of different data layouts (where data layouts are significant to both),
 *      then we need to perform necessary permutation op beforehand to unify their data layout before they can be fused with a post op
 *
 *      Note although post ops themselves should be able to support any data layout, the main op they fuse to may impose
 *      additional restrictions in the presence of post ops. For example, the implementation of a gemm op may only allow
 *      NHWC data layout if post ops are provided. Such restrictions are main op implementation specific.
 *
 *  @note: PostOps do not own any resources pointed to by TensorRelatedT if it's a pointer type
 *  @note: If TensorRelatedT points to a resource, IPostOp assumes that resource is valid throughout its lifetime
 *        and the lifetime of its copies. This is almost guaranteed as IPostOp is only meant to be used at configure time
 *        after the ITensor or ITensorInfo objects are already constructed
 */
template <typename TensorRelatedT>
struct IPostOp
{
    /** Get the arity of the post op
     * @note: that this is one fewer than the arity of the original op, because we implicitly pass the previous op's dst
     *       tensor as one of the arguments
     */
    size_t arity() const
    {
        return arguments().size();
    }
    /** The position of previous op's dst in current op's argument list */
    virtual int prev_dst_pos() const = 0;
    /** The IPostOp type */
    virtual PostOpType type() const = 0;
    /** The argument tensors
     * The order of the argument tensor is strictly preserved
     */
    virtual std::vector<TensorRelatedT *>       arguments()       = 0;
    virtual std::vector<const TensorRelatedT *> arguments() const = 0;
    /** Clone method used in cases where PostOps are owned by unique_ptr
     * @note: This performs a shallow copy of the TensorRelatedT if TensorRelatedT points to a resource
     */
    virtual std::unique_ptr<IPostOp<TensorRelatedT>> clone() const = 0;
    virtual ~IPostOp()
    {
    }
};

/** A sequence of PostOps that can be appended to the end of other operators */
template <typename TensorRelatedT>
class PostOpList
{
public:
    /** Constructor */
    PostOpList() = default;
    /** Destructor */
    ~PostOpList() = default;
    PostOpList(const PostOpList &other)
    {
        for(const auto &op : other._post_ops)
        {
            this->_post_ops.push_back(op->clone());
        }
    }
    PostOpList &operator=(const PostOpList &other)
    {
        PostOpList tmp{ other };
        std::swap(tmp, *this);
        return *this;
    }
    PostOpList(PostOpList &&other) = default;
    PostOpList &operator=(PostOpList &&other) = default;

    /** Add a new post op at the end of the list */
    template <typename OpT, typename... Args>
    void push_back_op(Args &&... args)
    {
        _post_ops.push_back(std::make_unique<OpT>(std::forward<Args>(args)...));
    }

    /** Number of post ops */
    size_t size() const
    {
        return _post_ops.size();
    }

    /** Total number of post ops */
    size_t total_num_arguments() const
    {
        return std::accumulate(_post_ops.begin(), _post_ops.end(), 0, [](size_t op1_arity, const auto & op2)
        {
            return op1_arity + op2->arity();
        });
    }

    /** Get the underlying post op list */
    std::vector<std::unique_ptr<IPostOp<TensorRelatedT>>> &get_list()
    {
        return _post_ops;
    }
    const std::vector<std::unique_ptr<IPostOp<TensorRelatedT>>> &get_list() const
    {
        return _post_ops;
    }

private:
    std::vector<std::unique_ptr<IPostOp<TensorRelatedT>>> _post_ops{};
};

} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_IPOSTOP
