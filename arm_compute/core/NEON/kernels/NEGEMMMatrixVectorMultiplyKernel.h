/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMMATRIXVECTORMULTIPLYKERNEL_H_
#define __ARM_COMPUTE_NEGEMMMATRIXVECTORMULTIPLYKERNEL_H_

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the GEMM matrix vector multiply kernel. **/
class NEGEMMMatrixVectorMultiplyKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMMatrixVectorMultiplyKernel";
    }
    /** Default constructor */
    NEGEMMMatrixVectorMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixVectorMultiplyKernel(const NEGEMMMatrixVectorMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixVectorMultiplyKernel &operator=(const NEGEMMMatrixVectorMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixVectorMultiplyKernel(NEGEMMMatrixVectorMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixVectorMultiplyKernel &operator=(NEGEMMMatrixVectorMultiplyKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input0 First Input tensor. Data types supported: QASYMM8/F16/F32
     * @param[in]  input1 Second Input tensor. Data types supported: same as @p input.
     * @param[out] output Output tensor which stores the interleaved matrix. Data type supported: same as @p input, S32 for QASYMM8 input.
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMMatrixVectorMultiplyKernel
     *
     * @param[in] input0 First Input tensor. Data types supported: QASYMM8/F16/F32
     * @param[in] input1 Second Input tensor. Data types supported: same as @p input.
     * @param[in] output Output tensor which stores the interleaved matrix. Data type supported: same as @p input, S32 for QASYMM8 input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Template function to run the matrix vector multiplication
     *
     * @tparam I0 Input 0 type
     * @tparam I1 Input 1 type
     * @tparam O  Output type
     *
     * @param[in] window_in  Input region. (Must be a valid region of the window returned by window()).
     * @param[in] window_w   Weights region. (Must be a valid region of the window returned by window()).
     * @param[in] window_out Output region.(Must be a valid region of the window returned by window()).
     */
    template <typename I0, typename I1, typename O>
    void matrix_vector_multiply(const Window &window_in, const Window &window_w, const Window &window_out);
    /** Common signature for all the specialised matrix vector multiplication functions */
    using GEMMMatrixVectorMultiplyFunctionPtr = void (NEGEMMMatrixVectorMultiplyKernel::*)(const Window &window_in,
                                                                                           const Window &window_w,
                                                                                           const Window &window_out);

private:
    GEMMMatrixVectorMultiplyFunctionPtr _func;
    const ITensor                      *_input0;
    const ITensor                      *_input1;
    ITensor                            *_output;
    BorderSize                          _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMMATRIXVECTORMULTIPLYKERNEL_H_*/
