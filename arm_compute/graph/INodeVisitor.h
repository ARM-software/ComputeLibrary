/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_INODEVISITOR_H__
#define __ARM_COMPUTE_GRAPH_INODEVISITOR_H__

#include "arm_compute/graph/nodes/NodesFwd.h"

namespace arm_compute
{
namespace graph
{
/**  Node visitor interface */
class INodeVisitor
{
public:
    /** Default destructor. */
    virtual ~INodeVisitor() = default;
    /** Visit INode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(INode &n) = 0;
    /** Visit ActivationLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(ActivationLayerNode &n) = 0;
    /** Visit BatchNormalizationLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(BatchNormalizationLayerNode &n) = 0;
    /** Visit ConcatenateLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(ConcatenateLayerNode &n) = 0;
    /** Visit ConstNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(ConstNode &n) = 0;
    /** Visit ConvolutionLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(ConvolutionLayerNode &n) = 0;
    /** Visit DepthwiseConvolutionLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(DepthwiseConvolutionLayerNode &n) = 0;
    /** Visit DetectionOutputLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(DetectionOutputLayerNode &n) = 0;
    /** Visit EltwiseLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(EltwiseLayerNode &n) = 0;
    /** Visit FlattenLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(FlattenLayerNode &n) = 0;
    /** Visit FullyConnectedLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(FullyConnectedLayerNode &n) = 0;
    /** Visit InputNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(InputNode &n) = 0;
    /** Visit NormalizationLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(NormalizationLayerNode &n) = 0;
    /** Visit OutputNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(OutputNode &n) = 0;
    /** Visit PermuteLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PermuteLayerNode &n) = 0;
    /** Visit PoolingLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PoolingLayerNode &n) = 0;
    /** Visit PriorBoxLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PriorBoxLayerNode &n) = 0;
    /** Visit ReshapeLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(ReshapeLayerNode &n) = 0;
    /** Visit SoftmaxLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(SoftmaxLayerNode &n) = 0;
    /** Visit SplitLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(SplitLayerNode &n) = 0;
};

/** Default visitor implementation
 *
 * Implements visit methods by calling a default function.
 * Inherit from DefaultNodeVisitor if you don't want to provide specific implementation for all nodes.
 */
class DefaultNodeVisitor : public INodeVisitor
{
public:
    /** Default destructor */
    virtual ~DefaultNodeVisitor() = default;

#ifndef DOXYGEN_SKIP_THIS
    // Inherited methods overridden
    virtual void visit(INode &n) override
    {
        default_visit();
    }
    virtual void visit(ActivationLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(BatchNormalizationLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(ConcatenateLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(ConstNode &n) override
    {
        default_visit();
    }
    virtual void visit(ConvolutionLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(DetectionOutputLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(DepthwiseConvolutionLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(EltwiseLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(FlattenLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(FullyConnectedLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(InputNode &n) override
    {
        default_visit();
    }
    virtual void visit(NormalizationLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(OutputNode &n) override
    {
        default_visit();
    }
    virtual void visit(PermuteLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(PoolingLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(PriorBoxLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(ReshapeLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(SoftmaxLayerNode &n) override
    {
        default_visit();
    }
    virtual void visit(SplitLayerNode &n) override
    {
        default_visit();
    }
#endif /* DOXYGEN_SKIP_THIS */

    /** Function to be overloaded by the client and implement default behavior for the
     *  non-overloaded visitors
     */
    virtual void default_visit() = 0;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_INODEVISITOR_H__ */
