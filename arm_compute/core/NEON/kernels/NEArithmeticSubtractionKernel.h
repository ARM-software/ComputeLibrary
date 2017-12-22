/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H__
#define __ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform subtraction between two tensors */
class NEArithmeticSubtractionKernel : public INEKernel
{
public:
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

    /** Initialise the kernel's input, output and border mode.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)     -> U8
     *   - (QS8,QS8)   -> QS8
     *   - (U8,U8)     -> S16
     *   - (S16,U8)    -> S16
     *   - (U8,S16)    -> S16
     *   - (S16,S16)   -> S16
     *   - (QS16,QS16) -> QS16
     *   - (F16,F16)   -> F16
     *   - (F32,F32)   -> F32
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in]  policy Overflow policy.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEArithmeticSubtractionKernel
     *
     * @param[in] input1 First tensor input. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in] input2 Second tensor input. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in] output Output tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised sub functions
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in]  input2 An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[out] output The output tensor. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in]  window Region on which to execute the kernel.
     */
    using SubFunction = void(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window);
    /** Sub function to use for the particular tensor types passed to configure() */
    SubFunction   *_func;
    const ITensor *_input1;
    const ITensor *_input2;
    ITensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEARITHMETICSUBTRACTIONKERNEL_H__ */
