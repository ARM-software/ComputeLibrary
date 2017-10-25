/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_IOPERATION_H__
#define __ARM_COMPUTE_GRAPH_IOPERATION_H__

#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Operation functor interface */
class IOperation
{
public:
    /** Virtual Destructor */
    virtual ~IOperation() = default;
    /** Interface to be implemented that configures an operation
     *
     * @param[in] ctx Node parameters to be used by the operation
     */
    virtual std::unique_ptr<arm_compute::IFunction> configure(NodeContext &ctx) = 0;
    /** Interface to be implemented that returns the target of the operation
     *
     * @return Target of the operation
     */
    virtual TargetHint target() const = 0;
};

#define REGISTER_SIMPLE_OPERATION(NAME, TARGET, OP)                                \
    class NAME : public IOperation                                                 \
    {                                                                              \
    public:                                                                    \
        std::unique_ptr<arm_compute::IFunction> configure(NodeContext &ctx) final; \
        TargetHint target() const final                                            \
        {                                                                          \
            return TargetHint::TARGET;                                             \
        }                                                                          \
    };                                                                             \
    static detail::OperationRegistrar<NAME> NAME##_registrar(OP);                  \
    std::unique_ptr<arm_compute::IFunction> NAME::configure(NodeContext &ctx)

} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_IOPERATION_H__ */
