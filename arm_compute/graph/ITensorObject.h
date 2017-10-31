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
#ifndef __ARM_COMPUTE_GRAPH_ITENSOROBJECT_H__
#define __ARM_COMPUTE_GRAPH_ITENSOROBJECT_H__

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Tensor object interface */
class ITensorObject
{
public:
    /** Default Destructor */
    virtual ~ITensorObject() = default;
    /** Calls accessor on tensor
     *
     * @return True if succeeds else false
     */
    virtual bool call_accessor() = 0;
    /** Checks if tensor has an accessor set.
     *
     * @return True if an accessor has been set else false
     */
    virtual bool has_accessor() const = 0;
    /** Sets target of the tensor
     *
     * @param[in] target Target where the tensor should be pinned in
     *
     * @return Backend tensor
     */
    virtual ITensor *set_target(TargetHint target) = 0;
    /** Returns a pointer to the internal tensor
     *
     * @return Tensor
     */
    virtual ITensor       *tensor()       = 0;
    virtual const ITensor *tensor() const = 0;
    /** Return the target that this tensor is pinned on
     *
     * @return Target of the tensor
     */
    virtual TargetHint target() const = 0;
    /** Allocates the tensor */
    virtual void allocate() = 0;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_ITENSOROBJECT_H__ */
