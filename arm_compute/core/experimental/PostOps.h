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
#ifndef ARM_COMPUTE_EXPERIMENTAL_POSTOPS
#define ARM_COMPUTE_EXPERIMENTAL_POSTOPS

#include "arm_compute/core/experimental/IPostOp.h"

#include "arm_compute/core/Types.h"

#include <vector>

namespace arm_compute
{
namespace experimental
{
/** (EXPERIMENTAL_POST_OPS)
 * Implementation of specific IPostOps
*/

template <typename TensorRelatedT>
struct PostOpAct : public IPostOp<TensorRelatedT>
{
public:
    PostOpAct(const ActivationLayerInfo &act_info)
        : _act_info{ act_info }
    {
    }
    // NOTE: PostOps do not own any resources pointed to by TensorRelatedT if it's a pointer type, thus allow shallow copy
    ~PostOpAct() override        = default;
    PostOpAct(const PostOpAct &) = default;
    PostOpAct &operator=(const PostOpAct &) = default;
    PostOpAct(PostOpAct &&)                 = default;
    PostOpAct &operator=(PostOpAct &&) = default;

    int prev_dst_pos() const override
    {
        return 0;
    }
    PostOpType type() const override
    {
        return PostOpType::Activation;
    }
    std::vector<TensorRelatedT *> arguments() override
    {
        return {};
    }
    std::vector<const TensorRelatedT *> arguments() const override
    {
        return {};
    }
    std::unique_ptr<IPostOp<TensorRelatedT>> clone() const override
    {
        return std::make_unique<PostOpAct<TensorRelatedT>>(*this);
    }
    ActivationLayerInfo _act_info;
};

template <typename TensorRelatedT>
struct PostOpEltwiseAdd : public IPostOp<TensorRelatedT>
{
public:
    PostOpEltwiseAdd(TensorRelatedT addend, int prev_dst_pos, ConvertPolicy policy)
        : _addend{ addend },
          _prev_dst_pos{ prev_dst_pos },
          _policy{ policy }
    {
    }
    // NOTE: PostOps do not own any resources pointed to by TensorRelatedT if it's a pointer type, thus allow shallow copy
    ~PostOpEltwiseAdd() override               = default;
    PostOpEltwiseAdd(const PostOpEltwiseAdd &) = default;
    PostOpEltwiseAdd &operator=(const PostOpEltwiseAdd &) = default;
    PostOpEltwiseAdd(PostOpEltwiseAdd &&)                 = default;
    PostOpEltwiseAdd &operator=(PostOpEltwiseAdd &&) = default;
    int               prev_dst_pos() const override
    {
        return _prev_dst_pos;
    }
    PostOpType type() const override
    {
        return PostOpType::Eltwise_Add;
    }
    std::vector<TensorRelatedT *> arguments() override
    {
        return { &_addend };
    }
    std::vector<const TensorRelatedT *> arguments() const override
    {
        return { &_addend };
    }
    std::unique_ptr<IPostOp<TensorRelatedT>> clone() const override
    {
        return std::make_unique<PostOpEltwiseAdd<TensorRelatedT>>(*this);
    }
    TensorRelatedT _addend;
    int            _prev_dst_pos;
    ConvertPolicy  _policy;
};

template <typename TensorRelatedT>
struct PostOpEltwisePRelu : public IPostOp<TensorRelatedT>
{
public:
    PostOpEltwisePRelu(TensorRelatedT alpha_param, int prev_dst_pos, ConvertPolicy policy)
        : _alpha_param{ alpha_param },
          _prev_dst_pos{ prev_dst_pos },
          _policy{ policy }
    {
    }
    // NOTE: PostOps do not own any resources pointed to by TensorRelatedT if it's a pointer type, thus allow shallow copy
    ~PostOpEltwisePRelu() override                 = default;
    PostOpEltwisePRelu(const PostOpEltwisePRelu &) = default;
    PostOpEltwisePRelu &operator=(const PostOpEltwisePRelu &) = default;
    PostOpEltwisePRelu(PostOpEltwisePRelu &&)                 = default;
    PostOpEltwisePRelu &operator=(PostOpEltwisePRelu &&) = default;
    int                 prev_dst_pos() const override
    {
        return _prev_dst_pos;
    }
    PostOpType type() const override
    {
        return PostOpType::Eltwise_PRelu;
    }
    std::vector<TensorRelatedT *> arguments() override
    {
        return { &_alpha_param };
    }
    std::vector<const TensorRelatedT *> arguments() const override
    {
        return { &_alpha_param };
    }
    std::unique_ptr<IPostOp<TensorRelatedT>> clone() const override
    {
        return std::make_unique<PostOpEltwisePRelu<TensorRelatedT>>(*this);
    }
    TensorRelatedT _alpha_param;
    int            _prev_dst_pos;
    ConvertPolicy  _policy;
};
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_POSTOPS