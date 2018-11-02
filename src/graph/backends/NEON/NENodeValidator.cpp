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
#include "arm_compute/graph/backends/NEON/NENodeValidator.h"

#include "arm_compute/graph/backends/ValidateHelpers.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

using namespace arm_compute::utils::cast;

namespace arm_compute
{
namespace graph
{
namespace backends
{
Status NENodeValidator::validate(INode *node)
{
    if(node == nullptr)
    {
        return Status{};
    }

    NodeType type = node->type();
    switch(type)
    {
        case NodeType::ConvolutionLayer:
            return detail::validate_convolution_layer<NEConvolutionLayer,
                   NEDirectConvolutionLayer,
                   NEGEMMConvolutionLayer,
                   NEWinogradConvolutionLayer>(*polymorphic_downcast<ConvolutionLayerNode *>(node));
        case NodeType::DepthwiseConvolutionLayer:
            return detail::validate_depthwise_convolution_layer<NEDepthwiseConvolutionLayer,
                   NEDepthwiseConvolutionLayer3x3>(*polymorphic_downcast<DepthwiseConvolutionLayerNode *>(node));

        default:
            return Status{};
    }
}
} // namespace backends
} // namespace graph
} // namespace arm_compute