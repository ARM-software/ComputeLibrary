/*
 * Copyright (c) 2018-2019,2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_ITENSOR_ACCESSOR_H
#define ARM_COMPUTE_GRAPH_ITENSOR_ACCESSOR_H

#include "arm_compute/core/ITensor.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Tensor accessor interface */
class ITensorAccessor
{
public:
    /** Default virtual destructor */
    virtual ~ITensorAccessor() = default;
    /** Interface to be implemented to access a given tensor
     *
     * @param[in] tensor Tensor to be accessed
     *
     * @return True if access is successful else false
     */
    virtual bool access_tensor(ITensor &tensor) = 0;
    /** Returns true if the tensor data is being accessed
     *
     * @return True if the tensor data is being accessed by the accessor. False otherwise
     */
    virtual bool access_tensor_data()
    {
        return true;
    }
};

using ITensorAccessorUPtr = std::unique_ptr<ITensorAccessor>;
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SUB_STREAM_H */