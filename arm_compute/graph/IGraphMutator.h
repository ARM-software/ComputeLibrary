/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_IGRAPHMUTATOR_H
#define ARM_COMPUTE_GRAPH_IGRAPHMUTATOR_H

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;

/** Graph mutator interface */
class IGraphMutator
{
public:
    /** Mutation type */
    enum class MutationType
    {
        IR,     /** IR specific mutation */
        Backend /** Backend specific mutation */
    };

public:
    /** Virtual Destructor */
    virtual ~IGraphMutator() = default;
    /** Walk the graph and perform a specific mutation
     *
     * @param[in, out] g Graph to walk and mutate
     */
    virtual void mutate(Graph &g) = 0;
    /** Returns mutation type
     *
     * @return Mutation type enumeration
     */
    virtual MutationType type() const = 0;
    /** Returns mutator name
     *
     * @return Mutator name
     */
    virtual const char *name() = 0;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_IGRAPHMUTATOR_H */
