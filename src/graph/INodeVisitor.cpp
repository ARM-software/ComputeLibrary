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
#include "arm_compute/graph/INodeVisitor.h"
#include "arm_compute/graph/nodes/Nodes.h"

namespace arm_compute
{
namespace graph
{
#ifndef DOXYGEN_SKIP_THIS
void DefaultNodeVisitor::visit(INode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(ActivationLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(BatchNormalizationLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(ConcatenateLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(ConstNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(ConvolutionLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(DequantizationLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(DetectionOutputLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(DetectionPostProcessLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(DepthwiseConvolutionLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(EltwiseLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(FlattenLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(FullyConnectedLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(FusedConvolutionBatchNormalizationNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(FusedConvolutionWithPostOpNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(FusedDepthwiseConvolutionBatchNormalizationNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(InputNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(NormalizationLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(OutputNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(PermuteLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(PoolingLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(PReluLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(PrintLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(PriorBoxLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(QuantizationLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(ReshapeLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(SoftmaxLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(SplitLayerNode &n)
{
    default_visit(n);
}
void DefaultNodeVisitor::visit(StackLayerNode &n)
{
    default_visit(n);
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace graph
} // namespace arm_compute