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
#ifndef __ARM_COMPUTE_GRAPH_INODE_H__
#define __ARM_COMPUTE_GRAPH_INODE_H__

#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Node interface */
class INode
{
public:
    /** Virtual Destructor */
    virtual ~INode() = default;
    /** Interface to be implemented that instantiates the node
     *
     * @param[in] hint   Hint to where the node should be executed
     * @param[in] input  Input tensor of the node
     * @param[in] output Output tensor of the node
     */
    virtual std::unique_ptr<arm_compute::IFunction> instantiate_node(Hint hint, ITensor *input, ITensor *output) = 0;
    /** Override the existing hint
     *
     * @note If the input is DONT_CARE then the method has to pick a technology,
     *       else it can accept the hint or override it (But not with DONT_CARE)
     *
     * @param[in] hint Hint to be considered
     *
     * @return The updated hint
     */
    Hint override_hint(Hint hint) const;

    virtual void print_info() = 0;

protected:
    /** Interface to be implement that override the hint
     *
     * @param[in] hint Hint to be considered
     *
     * @return The updated hint
     */
    virtual Hint node_override_hint(Hint hint) const;

protected:
    Hint     _hint{ Hint::DONT_CARE };
    ITensor *_input{ nullptr };
    ITensor *_output{ nullptr };
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_INODE_H__ */
