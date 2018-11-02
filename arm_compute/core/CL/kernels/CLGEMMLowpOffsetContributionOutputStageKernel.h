/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel used to add the offset contribution after @ref CLGEMMLowpMatrixMultiplyKernel and perform the output stage.
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), adds to it the offset contribution
 * of matrix A and matrix B and performs the output stage defined by the output_stage argument
 *
 */
class CLGEMMLowpOffsetContributionOutputStageKernel : public ICLKernel
{
public:
    /** Constructor */
    CLGEMMLowpOffsetContributionOutputStageKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpOffsetContributionOutputStageKernel(const CLGEMMLowpOffsetContributionOutputStageKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpOffsetContributionOutputStageKernel &operator=(const CLGEMMLowpOffsetContributionOutputStageKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpOffsetContributionOutputStageKernel(CLGEMMLowpOffsetContributionOutputStageKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpOffsetContributionOutputStageKernel &operator=(CLGEMMLowpOffsetContributionOutputStageKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mm_result      Input tensor containing the result of @ref CLGEMMLowpMatrixMultiplyKernel. Data type supported: S32
     * @param[in]  vector_sum_col Input row-vector of sums of all the entries in each column of matrix B.
     *                            Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  vector_sum_row Input row-vector of sums of all the entries in each row of matrix A.
     *                            Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  bias           Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                            Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output         Output tensor. Data type supported: QASYMM8
     * @param[in]  k              Number of matrix A columns or Matrix B rows
     * @param[in]  a_offset       Offset to be added to each element of the matrix A.
     * @param[in]  b_offset       Offset to be added to each element of the matrix B.
     * @param[in]  output_stage   GEMMLowp output stage info
     */
    void configure(const ICLTensor *mm_result, const ICLTensor *vector_sum_col, const ICLTensor *vector_sum_row, const ICLTensor *bias, ICLTensor *output, int32_t k, int32_t a_offset, int32_t b_offset,
                   const GEMMLowpOutputStageInfo &output_stage);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpOffsetContributionKernel
     *
     * @param[in] mm_result      Input tensor containing the result of @ref CLGEMMLowpOffsetContributionKernel. Data type supported: S32 or QASYMM8 if output_stage != NONE
     * @param[in] vector_sum_col Input row-vector of sums of all the entries in each column of matrix B.
     *                           Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in] vector_sum_row Input row-vector of sums of all the entries in each row of matrix A.
     *                           Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in] bias           Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                           Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output         Output tensor. Data type supported: QASYMM8
     * @param[in] a_offset       Offset to be added to each element of the matrix A.
     * @param[in] b_offset       Offset to be added to each element of the matrix B.
     * @param[in] output_stage   GEMMLowp output stage info
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output, int32_t a_offset,
                           int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_mm_result;
    const ICLTensor *_vector_sum_col;
    const ICLTensor *_vector_sum_row;
    const ICLTensor *_bias;
    ICLTensor       *_output;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__ */
