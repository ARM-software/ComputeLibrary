/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFILL_H
#define ARM_COMPUTE_NEFILL_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref cpu::kernels::CpuFillKernel */
class NEFill : public IFunction
{
public:
    /** Default Constructor */
    NEFill();
    /** Default Destructor */
    ~NEFill();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFill(const NEFill &) = delete;
    /** Default move constructor */
    NEFill(NEFill &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFill &operator=(const NEFill &) = delete;
    /** Default move assignment operator */
    NEFill &operator=(NEFill &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |All    |All    |
     *
     * @param[in,out] tensor         Source tensor. Data types supported: All
     * @param[in]     constant_value Constant value to use to fill tensor.
     */
    void configure(ITensor *tensor, PixelValue constant_value);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_FILL_H */
