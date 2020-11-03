/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFLOORKERNEL_H
#define ARM_COMPUTE_NEFLOORKERNEL_H

#include "src/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform a floor operation */
class NEFloorKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEFloorKernel";
    }
    /** Constructor */
    NEFloorKernel() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFloorKernel(const NEFloorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFloorKernel &operator=(const NEFloorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFloorKernel(NEFloorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFloorKernel &operator=(NEFloorKernel &&) = default;
    /** Default destructor */
    ~NEFloorKernel() = default;
    /** Set the source, destination of the kernel
     *
     * @param[in]  input  Source tensor. Data type supported: F16/F32.
     * @param[out] output Destination tensor. Same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFloorKernel
     *
     * @param[in] input  Source tensor info. Data type supported: F16/F32.
     * @param[in] output Destination tensor info. Same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEFLOORKERNEL_H */
