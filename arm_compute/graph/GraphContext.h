/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_CONTEXT_H__
#define __ARM_COMPUTE_GRAPH_CONTEXT_H__

#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** Hints that can be passed to the graph to expose parameterization */
class GraphHints
{
public:
    /** Default Constructor */
    GraphHints(TargetHint            target_hint      = TargetHint::DONT_CARE,
               ConvolutionMethodHint conv_method_hint = ConvolutionMethodHint::GEMM);
    /** Sets target execution hint
     *
     * @param target_hint Target execution hint
     */
    void set_target_hint(TargetHint target_hint);
    /** Sets convolution method to use
     *
     * @param convolution_method Convolution method to use
     */
    void set_convolution_method_hint(ConvolutionMethodHint convolution_method);
    /** Returns target execution hint
     *
     * @return target execution hint
     */
    TargetHint target_hint() const;
    /** Returns convolution method hint
     *
     * @return convolution method hint
     */
    ConvolutionMethodHint convolution_method_hint() const;

private:
    TargetHint            _target_hint;             /**< Target execution hint */
    ConvolutionMethodHint _convolution_method_hint; /**< Convolution method hint */
};

/** Graph context */
class GraphContext
{
public:
    /** Default Constuctor */
    GraphContext();
    /** Returns graph hints
     *
     * @return Graph hints
     */
    GraphHints &hints();
    /** Returns graph hints
     *
     * @return Graph hints
     */
    const GraphHints &hints() const;

private:
    GraphHints _hints; /**< Graph hints */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_CONTEXT_H__ */
