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
#ifndef ARM_COMPUTE_GRAPH_NODES_FWD_H
#define ARM_COMPUTE_GRAPH_NODES_FWD_H

namespace arm_compute
{
namespace graph
{
// Forward declarations
class INode;
class ActivationLayerNode;
class ArgMinMaxLayerNode;
class BatchNormalizationLayerNode;
class BoundingBoxTransformLayerNode;
class ChannelShuffleLayerNode;
class ConcatenateLayerNode;
class ConstNode;
class ConvolutionLayerNode;
class DeconvolutionLayerNode;
class DepthToSpaceLayerNode;
class DepthwiseConvolutionLayerNode;
class DequantizationLayerNode;
class DetectionOutputLayerNode;
class DetectionPostProcessLayerNode;
class DummyNode;
class EltwiseLayerNode;
class FlattenLayerNode;
class FullyConnectedLayerNode;
class FusedConvolutionBatchNormalizationNode;
class FusedConvolutionWithPostOpNode;
class FusedDepthwiseConvolutionBatchNormalizationNode;
class FusedConvolutionBatchNormalizationWithPostOpsNode;
class GenerateProposalsLayerNode;
class InputNode;
class L2NormalizeLayerNode;
class NormalizationLayerNode;
class NormalizePlanarYUVLayerNode;
class OutputNode;
class PadLayerNode;
class PermuteLayerNode;
class PoolingLayerNode;
class PReluLayerNode;
class PrintLayerNode;
class PriorBoxLayerNode;
class QuantizationLayerNode;
class ReductionLayerNode;
class ReorgLayerNode;
class ReshapeLayerNode;
class ResizeLayerNode;
class ROIAlignLayerNode;
class SoftmaxLayerNode;
class SliceLayerNode;
class SplitLayerNode;
class StackLayerNode;
class StridedSliceLayerNode;
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_NODES_FWD_H */
