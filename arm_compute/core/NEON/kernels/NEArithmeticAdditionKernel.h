/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEARITHMETICADDITIONKERNEL_H__
#define __ARM_COMPUTE_NEARITHMETICADDITIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform addition between two tensors */
class NEArithmeticAdditionKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEArithmeticAdditionKernel";
    }
    /** Default constructor */
    NEArithmeticAdditionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticAdditionKernel(const NEArithmeticAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticAdditionKernel &operator=(const NEArithmeticAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEArithmeticAdditionKernel(NEArithmeticAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEArithmeticAdditionKernel &operator=(NEArithmeticAdditionKernel &&) = default;
    /** Default destructor */
    ~NEArithmeticAdditionKernel() = default;

    /** Initialise the kernel's input, output and border mode.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QASYMM8/S16/F16/F32.
     * @param[in]  policy Overflow policy.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticAdditionKernel
     *
     * @param[in] input1 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[in] input2 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[in] output The output tensor. Data types supported: U8/QASYMM8/S16/F16/F32.
     * @param[in] policy Overflow policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised add functions
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QASYMM8/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QASYMM8/S16/F16/F32.
     * @param[in]  policy Overflow policy.
     * @param[in]  window Region on which to execute the kernel.
     */
    using AddFunction = void(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy, const Window &window);
    /** Add function to use for the particular tensor types passed to configure() */
    AddFunction   *_func;
    const ITensor *_input1;
    const ITensor *_input2;
    ITensor       *_output;
    ConvertPolicy  _policy;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEARITHMETICADDITIONKERNEL_H__ */
