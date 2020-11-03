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
#ifndef ARM_COMPUTE_NECOPYKERNEL_H
#define ARM_COMPUTE_NECOPYKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform a copy between two tensors */
class NECopyKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NECopyKernel";
    }
    /** Default constructor */
    NECopyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NECopyKernel(const NECopyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    NECopyKernel &operator=(const NECopyKernel &) = delete;
    /** Allow instances of this class to be moved */
    NECopyKernel(NECopyKernel &&) = default;
    /** Allow instances of this class to be moved */
    NECopyKernel &operator=(NECopyKernel &&) = default;
    /** Default destructor */
    ~NECopyKernel() = default;
    /** Initialize the kernel's input, output.
     *
     * @param[in]  input   Source tensor. Data types supported: All
     * @param[out] output  Destination tensor. Data types supported: same as @p input.
     * @param[in]  padding (Optional) Padding to be applied to the input tensor
     */
    void configure(const ITensor *input, ITensor *output, const PaddingList &padding = PaddingList());
    /** Static function to check if given info will lead to a valid configuration of @ref NECopyKernel
     *
     * @param[in] input   Source tensor. Data types supported: All
     * @param[in] output  Destination tensor. Data types supported: same as @p input.
     * @param[in] padding (Optional) Padding to be applied to the input tensor
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding = PaddingList());

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    PaddingList    _padding;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NECOPYKERNEL_H */
