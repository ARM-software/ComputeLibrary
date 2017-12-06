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
#ifndef __ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Common interface for all NEON reduction kernels */
class INEGEMMLowpReductionKernel : public INEKernel
{
public:
    /** Constructor */
    INEGEMMLowpReductionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    INEGEMMLowpReductionKernel(const INEGEMMLowpReductionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    INEGEMMLowpReductionKernel &operator=(const INEGEMMLowpReductionKernel &) = delete;
    /** Allow instances of this class to be moved */
    INEGEMMLowpReductionKernel(INEGEMMLowpReductionKernel &&) = default;
    /** Allow instances of this class to be moved */
    INEGEMMLowpReductionKernel &operator=(INEGEMMLowpReductionKernel &&) = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input       Input tensor. Data type supported: QASYMM8
     * @param[out] output      Output row-vector of sums of all the entries in each row/col of input tensor. Data type supported: S32
     * @param[in]  k           Number of matrix A columns (or matrix B rows)
     * @param[in]  is_reshaped True if the input tensor has been reshaped
     */
    virtual void configure(const ITensor *input, ITensor *output, int32_t k, bool is_reshaped) = 0;

protected:
    const ITensor *_input;
    ITensor       *_output;
    int32_t        _k;
    bool           _is_reshaped;
};

/** NEON kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class NEGEMMLowpMatrixAReductionKernel : public INEGEMMLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_a             Input tensor. Data type supported: QASYMM8
     * @param[out] vector_sum_row    Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in]  num_mtx_a_cols    Number of matrix A columns
     * @param[in]  is_interleaved4x4 True if the matrix A has been interleaved4x4
     */
    void configure(const ITensor *mtx_a, ITensor *vector_sum_row, int32_t num_mtx_a_cols, bool is_interleaved4x4) override;
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixAReductionKernel
     *
     * @param[in] mtx_a             Input tensor. Data type supported: QASYMM8
     * @param[in] vector_sum_row    Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in] num_mtx_a_cols    Number of matrix A columns
     * @param[in] is_interleaved4x4 True if the matrix A has been interleaved4x4
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, int32_t num_mtx_a_cols, bool is_interleaved4x4);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};

/** NEON kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class NEGEMMLowpMatrixBReductionKernel : public INEGEMMLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_b            Input tensor. Data type supported: Data type supported: QASYMM8
     * @param[out] vector_sum_col   Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in]  num_mtx_b_rows   Number of matrix B rows
     * @param[in]  is_transposed1xW True if the input tensor is transposed 1xW
     */
    void configure(const ITensor *mtx_b, ITensor *vector_sum_col, int32_t num_mtx_b_rows, bool is_transposed1xW) override;
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixBReductionKernel
     *
     * @param[in] mtx_b            Input tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] vector_sum_col   Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in] num_mtx_b_rows   Number of matrix B rows
     * @param[in] is_transposed1xW True if the input tensor is transposed 1xW
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, int32_t num_mtx_b_rows, bool is_transposed1xW);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H__ */
