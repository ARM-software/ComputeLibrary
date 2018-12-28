/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEREVERSEKERNEL_H__
#define __ARM_COMPUTE_NEREVERSEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the reverse layer kernel. */
class NEReverseKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEReverseKernel";
    }
    /** Default constructor */
    NEReverseKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReverseKernel(const NEReverseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReverseKernel &operator=(const NEReverseKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEReverseKernel(NEReverseKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEReverseKernel &operator=(NEReverseKernel &&) = default;
    /** Default destructor */
    ~NEReverseKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input  Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[out] output Output tensor. Data type supported: Same as @p input
     * @param[in]  axis   Axis tensor. Contains the indices of the dimensions to reverse. Data type supported: U32
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *axis);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReverseKernel
     *
     * @param[in] input  Input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in] output Output tensor info. Data type supported: Same as @p input
     * @param[in] axis   Axis tensor info. Contains the indices of the dimensions to reverse. Data type supported: U32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    const ITensor *_axis;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEREVERSEKERNEL_H__ */
