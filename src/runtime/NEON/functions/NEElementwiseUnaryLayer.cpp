/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"
#include "src/cpu/operators/CpuElementwiseUnary.h"
#include <utility>

namespace arm_compute
{
using OperatorType = cpu::CpuElementwiseUnary;

template <ElementWiseUnary op>
struct NEElementwiseUnaryLayer<op>::Impl
{
    const ITensor                *src{ nullptr };
    ITensor                      *dst{ nullptr };
    std::unique_ptr<OperatorType> cpu_op{ nullptr };
};

template <ElementWiseUnary op>
NEElementwiseUnaryLayer<op>::NEElementwiseUnaryLayer()
    : _impl(std::make_unique<Impl>())
{
}
template <ElementWiseUnary op>
NEElementwiseUnaryLayer<op>::~NEElementwiseUnaryLayer() = default;
template <ElementWiseUnary op>
NEElementwiseUnaryLayer<op>::NEElementwiseUnaryLayer(NEElementwiseUnaryLayer &&) = default;
template <ElementWiseUnary   op>
NEElementwiseUnaryLayer<op> &NEElementwiseUnaryLayer<op>::operator=(NEElementwiseUnaryLayer &&) = default;

template <ElementWiseUnary op>
void NEElementwiseUnaryLayer<op>::configure(const ITensor *input, ITensor *output)
{
    _impl->src    = input;
    _impl->dst    = output;
    _impl->cpu_op = std::make_unique<OperatorType>();
    _impl->cpu_op->configure(op, *_impl->src->info(), *_impl->dst->info());
}

template <ElementWiseUnary op>
Status NEElementwiseUnaryLayer<op>::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return OperatorType::validate(op, *input, *output);
}

template <ElementWiseUnary op>
void                       NEElementwiseUnaryLayer<op>::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->cpu_op->run(pack);
}

template class NEElementwiseUnaryLayer<ElementWiseUnary::RSQRT>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::EXP>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::NEG>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::LOG>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::ABS>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::ROUND>;
template class NEElementwiseUnaryLayer<ElementWiseUnary::SIN>;

} // namespace arm_compute
