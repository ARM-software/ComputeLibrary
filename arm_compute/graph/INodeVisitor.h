/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_INODEVISITOR_H
#define ARM_COMPUTE_GRAPH_INODEVISITOR_H

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
    /** Visit DequantizationLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(DequantizationLayerNode &n) = 0;
    /** Visit DetectionOutputLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(DetectionOutputLayerNode &n) = 0;
    /** Visit DetectionPostProcessLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(DetectionPostProcessLayerNode &n) = 0;
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
    /** Visit FusedConvolutionBatchNormalizationNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(FusedConvolutionBatchNormalizationNode &n) = 0;
    /** Visit FusedDepthwiseConvolutionBatchNormalizationNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(FusedDepthwiseConvolutionBatchNormalizationNode &n) = 0;
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
    /** Visit PreluLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PReluLayerNode &n) = 0;
    /** Visit PoolingLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PoolingLayerNode &n) = 0;
    /** Visit PrintLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PrintLayerNode &n) = 0;
    /** Visit PriorBoxLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(PriorBoxLayerNode &n) = 0;
    /** Visit QuantizationLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(QuantizationLayerNode &n) = 0;
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
    /** Visit StackLayerNode.
     *
     * @param[in] n Node to visit.
     */
    virtual void visit(StackLayerNode &n) = 0;
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
    virtual void visit(INode &) override
    {
        default_visit();
    }
    virtual void visit(ActivationLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(BatchNormalizationLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(ConcatenateLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(ConstNode &) override
    {
        default_visit();
    }
    virtual void visit(ConvolutionLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(DequantizationLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(DetectionOutputLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(DetectionPostProcessLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(DepthwiseConvolutionLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(EltwiseLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(FlattenLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(FullyConnectedLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(FusedConvolutionBatchNormalizationNode &) override
    {
        default_visit();
    }
    virtual void visit(FusedDepthwiseConvolutionBatchNormalizationNode &) override
    {
        default_visit();
    }
    virtual void visit(InputNode &) override
    {
        default_visit();
    }
    virtual void visit(NormalizationLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(OutputNode &) override
    {
        default_visit();
    }
    virtual void visit(PermuteLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(PoolingLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(PReluLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(PrintLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(PriorBoxLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(QuantizationLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(ReshapeLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(SoftmaxLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(SplitLayerNode &) override
    {
        default_visit();
    }
    virtual void visit(StackLayerNode &) override
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
#endif /* ARM_COMPUTE_GRAPH_INODEVISITOR_H */
