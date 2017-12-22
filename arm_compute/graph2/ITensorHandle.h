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
#ifndef __ARM_COMPUTE_GRAPH2_ITENSORHANDLE_H__
#define __ARM_COMPUTE_GRAPH2_ITENSORHANDLE_H__

#include "arm_compute/core/ITensor.h"

namespace arm_compute
{
namespace graph2
{
/** Tensor handle interface object **/
class ITensorHandle
{
public:
    /** Default virtual destructor **/
    virtual ~ITensorHandle() = default;
    /** Allocates backend memory for the handle **/
    virtual void allocate() = 0;
    /** Backend tensor object accessor **/
    virtual arm_compute::ITensor &tensor() = 0;
    /** Backend tensor object const accessor **/
    virtual const arm_compute::ITensor &tensor() const = 0;
    /** Maps backend tensor object
     *
     * @param[in] blocking Flags if the mapping operations should be blocking
     */
    virtual void map(bool blocking) = 0;
    /** Un-maps a backend tensor object **/
    virtual void unmap() = 0;
    /** Checks if a backing tensor is a sub-tensor object or not
     *
     * @return True if the backend tensor is a sub-tensor else false
     */
    virtual bool is_subtensor() const = 0;
};
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_ITENSORHANDLE_H__ */
