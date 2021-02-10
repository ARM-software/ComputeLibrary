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
#ifndef ARM_COMPUTE_NETILEKERNEL_H
#define ARM_COMPUTE_NETILEKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform a tile operation */
class NETileKernel : public INEKernel
{
public:
    /** Default constructor */
    NETileKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NETileKernel(const NETileKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NETileKernel &operator=(const NETileKernel &) = delete;
    /** Allow instances of this class to be moved */
    NETileKernel(NETileKernel &&) = default;
    /** Allow instances of this class to be moved */
    NETileKernel &operator=(NETileKernel &&) = default;
    /** Default destructor */
    ~NETileKernel() = default;
    const char *name() const override
    {
        return "NETileKernel";
    }
    /** Set the source, destination of the kernel
     *
     * @param[in]  input     Source tensor. Data type supported: All.
     * @param[out] output    Destination tensor. Same as @p input
     * @param[in]  multiples Contains the number of times the input tensor should be replicated on the given dimension.
     */
    void configure(const ITensor *input, ITensor *output, const Multiples &multiples);
    /** Static function to check if given info will lead to a valid configuration of @ref NETileKernel
     *
     * @param[in] input     Source tensor info. Data type supported: All.
     * @param[in] output    Destination tensor info. Same as @p input
     * @param[in] multiples Contains the number of times the input tensor should be replicated on the given dimension.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NETILEKERNEL_H */
