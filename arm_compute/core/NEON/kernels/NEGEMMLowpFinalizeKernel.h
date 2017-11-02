/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMLOWPFINALIZEKERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWPFINALIZEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/* NEON kernel used to finalize the GEMMLowp result
 *
 * This kernel performs the following computations:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result and round to nearest integer
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to uint8.
 */
class NEGEMMLowpFinalizeKernel : public INEKernel
{
public:
    /** Constructor */
    NEGEMMLowpFinalizeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpFinalizeKernel(const NEGEMMLowpFinalizeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpFinalizeKernel &operator=(const NEGEMMLowpFinalizeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpFinalizeKernel(NEGEMMLowpFinalizeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpFinalizeKernel &operator=(NEGEMMLowpFinalizeKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @note The input row-vectors  @p vector_sum_col and @p vector_sum_row must be the output of @ref NEGEMMLowpMatrixBReductionKernel and @ref NEGEMMLowpMatrixAReductionKernel kernels.
     *       These 2 vectors are needed to handle the offset of matrix product
     *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
     *
     * @param[in]  vector_sum_col Input row-vector of sums of all the entries in each column of input1.
     *                            Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: S32
     * @param[in]  vector_sum_row Input row-vector of sums of all the entries in each row of input0.
     *                            Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p vector_sum_col
     * @param[in]  mm_result      Input tensor containing the result of @ref NEGEMMLowpMatrixMultiplyKernel. Data type supported: same as @p vector_sum_col
     * @param[out] output         Output tensor containing the result of GEMMLowP. Data type supported: S8
     * @param[in]  num_mtx_a_cols Number of matrix A columns
     * @param[in]  a_offset       Offset to be added to each element of the matrix A.
     * @param[in]  b_offset       Offset to be added to each element of the matrix B.
     * @param[in]  c_offset       Offset to be added to each element of the output matrix
     * @param[in]  c_mult_int     Value to be multiplied to each entry of the result.
     * @param[in]  shift          Number of bits to shift right the result.
     */
    void configure(const ITensor *vector_sum_col, const ITensor *vector_sum_row, const ITensor *mm_result, ITensor *output, int32_t num_mtx_a_cols, int32_t a_offset, int32_t b_offset, int32_t c_offset,
                   int32_t c_mult_int, int32_t shift);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the finalize kernel
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <bool add_a_offset, bool add_b_offset>
    void finalize(const Window &window);
    using FinalizeFunctionPtr = void (NEGEMMLowpFinalizeKernel::*)(const Window &window);

    FinalizeFunctionPtr _func;
    const ITensor      *_vector_sum_col;
    const ITensor      *_vector_sum_row;
    const ITensor      *_mm_result;
    ITensor            *_output;
    int32_t             _a_offset;
    int32_t             _b_offset;
    int32_t             _c_offset;
    int32_t             _k_offset;
    int32_t             _c_mult_int;
    int32_t             _shift;
    bool                _slide_vector_sum_col;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_NEGEMMLOWPFINALIZEKERNEL_H__ */
