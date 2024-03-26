/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEBITWISEORKERNEL_H
#define ARM_COMPUTE_NEBITWISEORKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform bitwise inclusive OR between two tensors
 *
 * Result is computed by:
 * @f[ output(x,y) = input1(x,y) \lor input2(x,y) @f]
 */
class NEBitwiseOrKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBitwiseOrKernel";
    }
    /** Default constructor */
    NEBitwiseOrKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBitwiseOrKernel(const NEBitwiseOrKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBitwiseOrKernel &operator=(const NEBitwiseOrKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBitwiseOrKernel(NEBitwiseOrKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBitwiseOrKernel &operator=(NEBitwiseOrKernel &&) = default;
    /** Default destructor */
    ~NEBitwiseOrKernel() = default;
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  input1 An input tensor. Data type supported: U8.
     * @param[in]  input2 An input tensor. Data type supported: U8
     * @param[out] output Output tensor. Data type supported: U8.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input1; /**< Source tensor 1 */
    const ITensor *_input2; /**< Source tensor 2 */
    ITensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEBITWISEORKERNEL_H */
