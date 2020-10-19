/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NESIMPLEASSEMBLYFUNCTION_H
#define ARM_COMPUTE_NESIMPLEASSEMBLYFUNCTION_H

#include "arm_compute/runtime/IFunction.h"
#include "src/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"

#include <memory>

namespace arm_compute
{
/** Basic interface for functions which have a single NEON GEMM wrapper kernel to run */
class NESimpleAssemblyFunction : public IFunction
{
public:
    /** Constructor */
    NESimpleAssemblyFunction();

    /** Configure the function with the kernel to run
     *
     * @param[in] kernel GEMM Wrapper kernel configured and ready to run
     *
     * @note The kernel is expected to have a 1D window. The function will multi-thread this window across the X dimension.
     */
    void configure(std::unique_ptr<INEGEMMWrapperKernel> kernel);

    // Inherited methods overridden:
    void run() override final;

protected:
    std::unique_ptr<INEGEMMWrapperKernel> _kernel; /**< Kernel to run */
};
} //namespace arm_compute
#endif /*ARM_COMPUTE_NESIMPLEASSEMBLYFUNCTION_H */
