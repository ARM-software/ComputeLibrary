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
#ifndef ARM_COMPUTE_GRAPH_ISTREAM_OPERATORS_H
#define ARM_COMPUTE_GRAPH_ISTREAM_OPERATORS_H

#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph/frontend/Types.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
// Forward declarations
class ILayer;

/** Overloaded stream operator to add a node to the graph
 *
 * @param[in, out] s     Stream to add the tensor
 * @param[in]      layer Layer to be added
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, ILayer &&layer)
{
    s.add_layer(layer);
    return s;
}
/** Overloaded stream operator to add a node to the graph
 *
 * @param[in, out] s     Stream to add the tensor
 * @param[in]      layer Layer to be added
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, ILayer &layer)
{
    s.add_layer(layer);
    return s;
}
/** Overloaded stream operator to provide a target hint to the graph
 *
 * @param[in, out] s           Stream to provide the hint to
 * @param[in]      target_hint Target hint to be considered
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, Target target_hint)
{
    s.hints().target_hint = target_hint;
    return s;
}
/** Overloaded stream operator to provide a convolution method hint to the graph
 *
 * @param[in, out] s                       Stream to provide the hint to
 * @param[in]      convolution_method_hint Convolution method hint to be considered
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, ConvolutionMethod convolution_method_hint)
{
    s.hints().convolution_method_hint = convolution_method_hint;
    return s;
}
/** Overloaded stream operator to provide a depthwise convolution method hint to the graph
 *
 * @param[in, out] s                                 Stream to provide the hint to
 * @param[in]      depthwise_convolution_method_hint Depthwise Convolution method hint to be considered
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, DepthwiseConvolutionMethod depthwise_convolution_method_hint)
{
    s.hints().depthwise_convolution_method_hint = depthwise_convolution_method_hint;
    return s;
}
/** Overloaded stream operator to provide a fast math hint to the graph
 *
 * @param[in, out] s              Stream to provide the hint to
 * @param[in]      fast_math_hint Convolution method hint to be considered
 *
 * @return Updated stream
 */
inline IStream &operator<<(IStream &s, FastMathHint fast_math_hint)
{
    s.hints().fast_math_hint = fast_math_hint;
    return s;
}
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ISTREAM_OPERATORS_H */
