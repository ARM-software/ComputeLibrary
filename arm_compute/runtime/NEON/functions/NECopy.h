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
#ifndef ARM_COMPUTE_NECOPY_H
#define ARM_COMPUTE_NECOPY_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuCopyKernel */
class NECopy : public IFunction
{
public:
    /** Default Constructor */
    NECopy();
    /** Default Destructor */
    ~NECopy();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECopy(const NECopy &) = delete;
    /** Default move constructor */
    NECopy(NECopy &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECopy &operator=(const NECopy &) = delete;
    /** Default move assignment operator */
    NECopy &operator=(NECopy &&);
    /** Initialise the function's source and destination.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input  Source tensor. Data types supported: All
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NECopy
     *
     * @param[in] input  Source tensor. Data types supported: All
     * @param[in] output Output tensor. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NECOPY_H */
