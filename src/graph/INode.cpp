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

#include "arm_compute/graph/INode.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Validate.h"

#include <ostream>

using namespace arm_compute::graph;

Hint INode::override_hint(Hint hint) const
{
    if(hint == Hint::OPENCL && !opencl_is_available())
    {
        hint = Hint::DONT_CARE;
    }
    hint = node_override_hint(hint);
    ARM_COMPUTE_ERROR_ON(hint == Hint::OPENCL && !opencl_is_available());
    return hint;
}
Hint INode::node_override_hint(Hint hint) const
{
    return hint == Hint::DONT_CARE ? Hint::NEON : hint;
}
