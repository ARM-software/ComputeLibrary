/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H
#define ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
struct GEMMLowpReductionKernelInfo;

/** Common interface for all Neon reduction kernels */
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
    /** Default destructor */
    virtual ~INEGEMMLowpReductionKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] output Output row-vector of sums of all the entries in each row/col of input tensor. Data type supported: S32
     * @param[in]  info   Kernel metadata:
     *                    - k            Number of matrix columns/rows depending on the type of reduction.
     *                    - is_reshaped  True if the matrix has been reshaped.
     *                    - scalar       Scalar value to multiply each reduced column/row by.
     *                    - mul_byscalar True if each reduced column/row must be multiplied by a scalar value.
     */
    virtual void configure(const ITensor *input, ITensor *output, const GEMMLowpReductionKernelInfo &info) = 0;

protected:
    const ITensor *_input;
    ITensor       *_output;
    int32_t        _k;
    int32_t        _scalar;
    bool           _mul_by_scalar;
};

/** Neon kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class NEGEMMLowpMatrixAReductionKernel : public INEGEMMLowpReductionKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMLowpMatrixAReductionKernel";
    }
    /** Default constructor */
    NEGEMMLowpMatrixAReductionKernel() = default;
    /** Prevent instances of this class from being copied */
    NEGEMMLowpMatrixAReductionKernel(const NEGEMMLowpMatrixAReductionKernel &) = delete;
    /** Prevent instances of this class from being copied */
    NEGEMMLowpMatrixAReductionKernel &operator=(const NEGEMMLowpMatrixAReductionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixAReductionKernel(NEGEMMLowpMatrixAReductionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixAReductionKernel &operator=(NEGEMMLowpMatrixAReductionKernel &&) = default;
    /** Default destructor */
    ~NEGEMMLowpMatrixAReductionKernel() = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_a          Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] vector_sum_row Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in]  info           Kernel metadata:
     *                            - k            (num_mtx_a_cols) Number of matrix A columns
     *                            - is_reshaped  (is_interleaved4x4) True if the matrix A has been interleaved4x4
     *                            - scalar       Scalar value to multiply each reduced row by.
     *                            - mul_byscalar True if each reduced column must be multiplied by a scalar value.
     */
    void configure(const ITensor *mtx_a, ITensor *vector_sum_row, const GEMMLowpReductionKernelInfo &info) override;
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixAReductionKernel
     *
     * @param[in] mtx_a          Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[in] vector_sum_row Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in] info           Kernel metadata:
     *                           - k            (num_mtx_a_cols) Number of matrix A columns
     *                           - is_reshaped  (is_interleaved4x4) True if the matrix A has been interleaved4x4
     *                           - scalar       Scalar value to multiply each reduced row by.
     *                           - mul_byscalar True if each reduced column must be multiplied by a scalar value.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Execution of the reduction kernel specialized on the input type
     *
     * @param[in] window Execution window
     */
    template <typename T>
    void run_internal(const Window &window);
};

/** Neon kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class NEGEMMLowpMatrixBReductionKernel : public INEGEMMLowpReductionKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMLowpMatrixBReductionKernel";
    }
    /** Default constructor */
    NEGEMMLowpMatrixBReductionKernel() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpMatrixBReductionKernel(const NEGEMMLowpMatrixBReductionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpMatrixBReductionKernel &operator=(const NEGEMMLowpMatrixBReductionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixBReductionKernel(NEGEMMLowpMatrixBReductionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixBReductionKernel &operator=(NEGEMMLowpMatrixBReductionKernel &&) = default;
    /** Default destructor */
    ~NEGEMMLowpMatrixBReductionKernel() = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_b          Input tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] vector_sum_col Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in]  info           Kernel metadata:
     *                            - k            (num_mtx_b_rows) Number of matrix B rows.
     *                            - is_reshaped  (is_transposed1xW) True if the input tensor is transposed 1xW.
     *                            - scalar       Scalar value to multiply each reduced row by.
     *                            - mul_byscalar True if each reduced row must be multiplied by a scalar value.
     */
    void configure(const ITensor *mtx_b, ITensor *vector_sum_col, const GEMMLowpReductionKernelInfo &info) override;
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixBReductionKernel
     *
     * @param[in] mtx_b          Input tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[in] vector_sum_col Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in] info           Kernel metadata:
     *                           - k            (num_mtx_b_rows) Number of matrix B rows.
     *                           - is_reshaped  (is_transposed1xW) True if the input tensor is transposed 1xW.
     *                           - scalar       Scalar value to multiply each reduced row by.
     *                           - mul_byscalar True if each reduced row must be multiplied by a scalar value.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Execution of the reduction kernel specialized on the input type
     *
     * @param[in] window Execution window
     * @param[in] info   Thread-related information
     */
    template <typename T>
    void run_internal(const Window &window, const ThreadInfo &info);
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEGEMMLOWREDUCTIONKERNEL_H */
