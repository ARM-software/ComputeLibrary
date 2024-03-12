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
#ifndef ARM_COMPUTE_GRAPH_ITENSORHANDLE_H
#define ARM_COMPUTE_GRAPH_ITENSORHANDLE_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
// Forward declarations
class IMemoryGroup;

namespace graph
{
/** Tensor handle interface object */
class ITensorHandle
{
public:
    /** Default virtual destructor */
    virtual ~ITensorHandle() = default;
    /** Allocates backend memory for the handle */
    virtual void allocate() = 0;
    /** Allocates backend memory for the handle */
    virtual void free() = 0;
    /** Set backend tensor to be managed by a memory group
     *
     * @param[in] mg Memory group
     */
    virtual void manage(IMemoryGroup *mg) = 0;
    /** Maps backend tensor object
     *
     * @param[in] blocking Flags if the mapping operations should be blocking
     */
    virtual void map(bool blocking) = 0;
    /** Un-maps a backend tensor object */
    virtual void unmap() = 0;
    /** Releases backend tensor if is marked as unused
     *
     *
     * @note This has no effect on sub-tensors
     * @warning Parent tensors don't keep track of sub-tensors,
     *          thus if a parent is set as unused then all sub-tensors will be invalidated,
     *          on the other hand if a sub-tensor is marked as unused then the parent tensor won't be released
     */
    virtual void release_if_unused() = 0;
    /** Backend tensor object accessor */
    virtual arm_compute::ITensor &tensor() = 0;
    /** Backend tensor object const accessor */
    virtual const arm_compute::ITensor &tensor() const = 0;
    /** Return the parent tensor handle if is a subtensor else this
     *
     * @return Parent tensor handle
     */
    virtual ITensorHandle *parent_handle() = 0;
    /** Checks if a backing tensor is a sub-tensor object or not
     *
     * @return True if the backend tensor is a sub-tensor else false
     */
    virtual bool is_subtensor() const = 0;
    /** Returns target type
     *
     * @return Target type
     */
    virtual Target target() const = 0;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ITENSORHANDLE_H */
