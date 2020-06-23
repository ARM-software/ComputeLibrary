/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H
#define ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform subtraction between two tensors */
class NEArithmeticSubtractionKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEArithmeticSubtractionKernel";
    }
    /** Default constructor */
    NEArithmeticSubtractionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticSubtractionKernel(const NEArithmeticSubtractionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEArithmeticSubtractionKernel &operator=(const NEArithmeticSubtractionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEArithmeticSubtractionKernel(NEArithmeticSubtractionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEArithmeticSubtractionKernel &operator=(NEArithmeticSubtractionKernel &&) = default;
    /** Default destructor */
    ~NEArithmeticSubtractionKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)                          -> U8
     *   - (U8,U8)                          -> S16
     *   - (QASYMM8, QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED, QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (S16,U8)                         -> S16
     *   - (U8,S16)                         -> S16
     *   - (S16,S16)                        -> S16
     *   - (F16,F16)                        -> F16
     *   - (F32,F32)                        -> F32
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32.
     * @param[in]  policy Overflow policy. Convert policy cannot be WRAP if datatype is quantized.
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticSubtractionKernel
     *
     * @note Convert policy cannot be WRAP if datatype is QASYMM8
     *
     * @param[in] input1 An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in] input2 An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in] output The output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32.
     * @param[in] policy Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden:
    void run_op(const InputTensorMap &inputs, const OutputTensorMap &outputs, const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised sub functions
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/F16/F32.
     * @param[in]  window Region on which to execute the kernel.
     * @param[in]  is_sat Flag to indicate if the policy is SATURATE.
     */
    using SubFunction = void(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window, bool is_sat);
    /** Sub function to use for the particular tensor types passed to configure() */
    SubFunction *_func;
    ConvertPolicy _policy;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H */
