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
#ifndef ARM_COMPUTE_GRAPH_NEFUNCTIONFACTORY_H
#define ARM_COMPUTE_GRAPH_NEFUNCTIONFACTORY_H

#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
// Forward declarations
class INode;
class GraphContext;

namespace backends
{
/** Factory for generating Neon backend functions **/
class NEFunctionFactory final
{
public:
    /** Create a backend execution function depending on the node type
     *
     * @param[in] node Node to create the backend function for
     * @param[in] ctx  Context to use
     *
     * @return Backend function
     */
    static std::unique_ptr<arm_compute::IFunction> create(INode *node, GraphContext &ctx);
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif //ARM_COMPUTE_GRAPH_NEFUNCTIONFACTORY_H
