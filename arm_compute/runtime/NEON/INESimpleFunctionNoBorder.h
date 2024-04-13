/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H
#define ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"

#include <memory>

namespace arm_compute
{
class ICPPKernel;
using INEKernel = ICPPKernel;
/** Basic interface for functions which have a single CPU kernel and no border */
class INESimpleFunctionNoBorder : public IFunction
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    INESimpleFunctionNoBorder(IRuntimeContext *ctx = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INESimpleFunctionNoBorder(const INESimpleFunctionNoBorder &) = delete;
    /** Default move constructor */
    INESimpleFunctionNoBorder(INESimpleFunctionNoBorder &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INESimpleFunctionNoBorder &operator=(const INESimpleFunctionNoBorder &) = delete;
    /** Default move assignment operator */
    INESimpleFunctionNoBorder &operator=(INESimpleFunctionNoBorder &&) = default;
    /** Default destructor */
    ~INESimpleFunctionNoBorder();

    // Inherited methods overridden:
    void run() override final;

protected:
    std::unique_ptr<INEKernel> _kernel; /**< Kernel to run */
    IRuntimeContext           *_ctx;    /**< Context to use */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H */
