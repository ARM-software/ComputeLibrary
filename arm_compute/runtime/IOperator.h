/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_IOPERATOR_H
#define ARM_COMPUTE_IOPERATOR_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Types.h"

namespace arm_compute
{
class ITensorPack;
namespace experimental
{
/** Base class for all async functions */
class IOperator
{
public:
    /** Destructor */
    virtual ~IOperator() = default;
    /** Run the kernels contained in the function
     *
     * @param[in] tensors Vector that contains the tensors to operate on.
     *
     */
    virtual void run(ITensorPack &tensors) = 0;
    /** Prepare the function for executing
     *
     * Any one off pre-processing step required by the function is handled here
     *
     * @param[in] constants Vector that contains the constants tensors.
     *
     * @note Prepare stage might not need all the function's buffers' backing memory to be available in order to execute
     */
    virtual void prepare(ITensorPack &constants) = 0;

    /** Return the memory requirements required by the workspace
     */
    virtual MemoryRequirements workspace() const = 0;
};
} // namespace experimental
} // namespace arm_compute
#endif /*ARM_COMPUTE_IOPERATOR_H */
