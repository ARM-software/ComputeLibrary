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
#ifndef ARM_COMPUTE_ICLOPERATOR_H
#define ARM_COMPUTE_ICLOPERATOR_H

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Types.h"

#include <memory>

namespace arm_compute
{
class ICLKernel;
namespace experimental
{
/** Basic interface for functions which have a single async CL kernel */
class ICLOperator : public IOperator
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    ICLOperator(IRuntimeContext *ctx = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLOperator(const ICLOperator &) = delete;
    /** Default move constructor */
    ICLOperator(ICLOperator &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLOperator &operator=(const ICLOperator &) = delete;
    /** Default move assignment operator */
    ICLOperator &operator=(ICLOperator &&) = default;

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    MemoryRequirements workspace() const override;

protected:
    std::unique_ptr<ICLKernel> _kernel;
    IRuntimeContext           *_ctx;
    MemoryRequirements         _workspace;
};
} // namespace experimental
} // namespace arm_compute
#endif /*ARM_COMPUTE_ICLOPERATOR_H */
