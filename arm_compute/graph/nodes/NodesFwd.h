/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_NODES_FWD_H__
#define __ARM_COMPUTE_GRAPH_NODES_FWD_H__

namespace arm_compute
{
namespace graph
{
// Forward declarations
class INode;
class ActivationLayerNode;
class BatchNormalizationLayerNode;
class BoundingBoxTransformLayerNode;
class ChannelShuffleLayerNode;
class ConcatenateLayerNode;
class ConstNode;
class ConvolutionLayerNode;
class DeconvolutionLayerNode;
class DepthwiseConvolutionLayerNode;
class DetectionOutputLayerNode;
class DetectionPostProcessLayerNode;
class DummyNode;
class EltwiseLayerNode;
class FlattenLayerNode;
class FullyConnectedLayerNode;
class FusedConvolutionBatchNormalizationNode;
class FusedDepthwiseConvolutionBatchNormalizationNode;
class GenerateProposalsLayerNode;
class InputNode;
class NormalizationLayerNode;
class NormalizePlanarYUVLayerNode;
class OutputNode;
class PadLayerNode;
class PermuteLayerNode;
class PoolingLayerNode;
class PriorBoxLayerNode;
class QuantizationLayerNode;
class ReorgLayerNode;
class ReshapeLayerNode;
class ResizeLayerNode;
class ROIAlignLayerNode;
class SoftmaxLayerNode;
class SliceLayerNode;
class SplitLayerNode;
class StackLayerNode;
class UpsampleLayerNode;
class YOLOLayerNode;
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_NODES_FWD_H__ */
