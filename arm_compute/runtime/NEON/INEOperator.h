/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_INEOPERATOR_H
#define ARM_COMPUTE_INEOPERATOR_H

#include "../../core/ITensor.h"
#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Types.h"

#include <memory>

namespace arm_compute
{
class ICPPKernel;
class Window;

using INEKernel = ICPPKernel;
namespace experimental
{
/** Basic interface for functions which have a single async CPU kernel */
class INEOperator : public IOperator
{
public:
    /** Constructor
     *
     * @param[in] ctx Runtime context to be used by the function
     */
    INEOperator(IRuntimeContext *ctx = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEOperator(const INEOperator &) = delete;
    /** Default move constructor */
    INEOperator(INEOperator &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    INEOperator &operator=(const INEOperator &) = delete;
    /** Default move assignment operator */
    INEOperator &operator=(INEOperator &&) = default;
    /** Default destructor */
    ~INEOperator();

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    MemoryRequirements workspace() const override;

protected:
    void run(ITensorPack &tensors, const Window &window);

    std::unique_ptr<INEKernel> _kernel;
    IRuntimeContext           *_ctx;
    MemoryRequirements         _workspace;
};
} // namespace experimental
} // namespace arm_compute
#endif /*ARM_COMPUTE_INEOPERATOR_H */
