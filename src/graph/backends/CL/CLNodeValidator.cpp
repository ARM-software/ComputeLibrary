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
#include "arm_compute/graph/backends/CL/CLNodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
Status CLNodeValidator::validate(INode *node)
{
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ChannelShuffleLayer:
            return detail::validate_channel_shuffle_layer<CLChannelShuffleLayer>(*polymorphic_downcast<ChannelShuffleLayerNode *>(node));
        case NodeType::ConvolutionLayer:
            return detail::validate_convolution_layer<CLConvolutionLayer,
                   CLDirectConvolutionLayer,
                   CLGEMMConvolutionLayer,
                   CLWinogradConvolutionLayer>(*polymorphic_downcast<ConvolutionLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::validate_depthwise_convolution_layer<CLDepthwiseConvolutionLayer,
                   CLDepthwiseConvolutionLayer3x3>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));
        case NodeType::NormalizePlanarYUVLayer:
            return detail::validate_normalize_planar_yuv_layer<CLNormalizePlanarYUVLayer>(*polymorphic_downcast<NormalizePlanarYUVLayerNode *>(node));
        case NodeType::PermuteLayer:
            return detail::validate_permute_layer<CLPermute>(*polymorphic_downcast<PermuteLayerNode *>(node));
        case NodeType::ReorgLayer:
            return detail::validate_reorg_layer<CLReorgLayer>(*polymorphic_downcast<ReorgLayerNode *>(node));
        default:
            return Status{};
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
