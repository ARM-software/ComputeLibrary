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
#ifndef __ARM_COMPUTE_CLGEMMLOWREDUCTIONKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWREDUCTIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Common interface for all OpenCL reduction kernels */
class ICLGEMMLowpReductionKernel : public ICLKernel
{
public:
    /** Constructor */
    ICLGEMMLowpReductionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    ICLGEMMLowpReductionKernel(const ICLGEMMLowpReductionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    ICLGEMMLowpReductionKernel &operator=(const ICLGEMMLowpReductionKernel &) = delete;
    /** Allow instances of this class to be moved */
    ICLGEMMLowpReductionKernel(ICLGEMMLowpReductionKernel &&) = default;
    /** Allow instances of this class to be moved */
    ICLGEMMLowpReductionKernel &operator=(ICLGEMMLowpReductionKernel &&) = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data type supported: S8
     * @param[out] output Output row-vector of sums of all the entries in each row/col of input tensor. Data type supported: S32
     */
    virtual void configure(const ICLTensor *input, ICLTensor *output) = 0;

protected:
    const ICLTensor *_input;
    ICLTensor       *_output;
};

/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class CLGEMMLowpMatrixAReductionKernel : public ICLGEMMLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_a          Input tensor. Data type supported: QASYMM8
     * @param[out] vector_sum_row Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     */
    void configure(const ICLTensor *mtx_a, ICLTensor *vector_sum_row) override;
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixAReductionKernel
     *
     * @param[in] mtx_a          Input tensor. Data type supported: QASYMM8
     * @param[in] vector_sum_row Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
};

/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class CLGEMMLowpMatrixBReductionKernel : public ICLGEMMLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mtx_b          Input tensor. Data type supported: Data type supported: QASYMM8
     * @param[out] vector_sum_col Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     */
    void configure(const ICLTensor *mtx_b, ICLTensor *vector_sum_col) override;
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixBReductionKernel
     *
     * @param[in] mtx_b          Input tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] vector_sum_col Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_CLGEMMLOWREDUCTIONKERNEL_H__ */
