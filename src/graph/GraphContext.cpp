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
#include "arm_compute/graph/GraphContext.h"

using namespace arm_compute::graph;

GraphHints::GraphHints(TargetHint target_hint, ConvolutionMethodHint conv_method_hint)
    : _target_hint(target_hint), _convolution_method_hint(conv_method_hint)
{
}

void GraphHints::set_target_hint(TargetHint target_hint)
{
    _target_hint = target_hint;
}

void GraphHints::set_convolution_method_hint(ConvolutionMethodHint convolution_method)
{
    _convolution_method_hint = convolution_method;
}

TargetHint GraphHints::target_hint() const
{
    return _target_hint;
}

ConvolutionMethodHint GraphHints::convolution_method_hint() const
{
    return _convolution_method_hint;
}

GraphContext::GraphContext()
    : _hints()
{
}

GraphHints &GraphContext::hints()
{
    return _hints;
}

const GraphHints &GraphContext::hints() const
{
    return _hints;
}