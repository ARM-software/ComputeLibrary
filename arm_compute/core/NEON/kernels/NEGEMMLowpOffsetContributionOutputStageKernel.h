/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel used to add the offset contribution and perform the output stage after @ref NEGEMMLowpMatrixMultiplyKernel.
 *
 * The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyKernel),
 * and adds to it the offset contribution of matrix A and matrix B in-place.
 *
 * The output stage can perform either QuantizeDownInt32ToUint8Scale or QuantizeDownInt32ToUint8ScaleByFixedPoint.
 *
 * For QuantizeDownInt32ToUint8Scale the final result is:
 *
 * ((mm_result'[i][k] + result_offset) * result_mult_int) >> result_shift
 *
 * For QuantizeDownInt32ToUint8ScaleByFixedPoint the final result is:
 *
 * (FixedPointMul(mm_result'[i][k], result_fixedpoint_multiplier) >> result_shift) + result_offset_after_shift
 *
 * where FixedPointMul(x, y) is the nearest integer to the following
 * mathematical expression, evaluated without overflow or intermediate rounding:
 *
 * (x * y) / 2^31
 *
 * and mm_result'[i][k] = mm_result[i][k] +
 *                        (vector_sum_col[k] * a_offset) +
 *                        (vector_sum_row[i] * b_offset) +
 *                        (a_offset * b_offset * k)
 */

class NEGEMMLowpOffsetContributionOutputStageKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMLowpOffsetContributionOutputStageKernel";
    }
    /** Constructor */
    NEGEMMLowpOffsetContributionOutputStageKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpOffsetContributionOutputStageKernel(const NEGEMMLowpOffsetContributionOutputStageKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpOffsetContributionOutputStageKernel &operator=(const NEGEMMLowpOffsetContributionOutputStageKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpOffsetContributionOutputStageKernel(NEGEMMLowpOffsetContributionOutputStageKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpOffsetContributionOutputStageKernel &operator=(NEGEMMLowpOffsetContributionOutputStageKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  mm_result      Input tensor containing the result of @ref NEGEMMLowpMatrixMultiplyKernel. Data type supported: S32
     * @param[in]  vector_sum_col Input row-vector of sums of all the entries in each column of matrix B.
     *                            Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  vector_sum_row Input row-vector of sums of all the entries in each row of matrix A.
     * @param[in]  bias           Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                            Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p mm_result.
     * @param[out] output         Output tensor containing the final quantized result. Data type supported: QASYMM8
     * @param[in]  k              Number of matrix A columns or Matrix B rows
     * @param[in]  a_offset       Offset to be added to each element of the matrix A.
     * @param[in]  b_offset       Offset to be added to each element of the matrix B.
     * @param[in]  output_stage   GEMMLowp output stage info, providing the type of quantization and the necessary parameters.
     */
    void configure(const ITensor *mm_result, const ITensor *vector_sum_col, const ITensor *vector_sum_row, const ITensor *bias, ITensor *output, int32_t k, int32_t a_offset, int32_t b_offset,
                   GEMMLowpOutputStageInfo output_stage);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpOffsetContributionOutputStageKernel
     *
     * @param[in] mm_result      Input tensor info containing the result of @ref NEGEMMLowpMatrixMultiplyKernel. Data type supported: S32
     * @param[in] vector_sum_col Tensor info for the input row-vector of sums of all the entries in each column of matrix B.
     *                           Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in] vector_sum_row Tensor info for the input row-vector of sums of all the entries in each row of matrix A.
     *                           Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in] bias           Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                           Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p mm_result.
     * @param[in] output         Output tensor info containing the final quantized result. Data type supported: QASYMM8
     * @param[in] a_offset       Offset to be added to each element of the matrix A.
     * @param[in] b_offset       Offset to be added to each element of the matrix B.
     * @param[in] output_stage   GEMMLowp output stage info, providing the type of quantization and the necessary parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output, int32_t a_offset,
                           int32_t                 b_offset,
                           GEMMLowpOutputStageInfo output_stage);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    using NEGEMMLowpOffsetContributionOutputStageFunction = std::function<void(const Window, const ITensor *, const ITensor *, const ITensor *, const ITensor *,
                                                                               ITensor *, int32_t, int32_t, int32_t, bool, GEMMLowpOutputStageInfo)>;

private:
    /** Function to use for the particular tensors passed to configure() */
    NEGEMMLowpOffsetContributionOutputStageFunction _function;
    const ITensor                                  *_vector_sum_col;
    const ITensor                                  *_vector_sum_row;
    const ITensor                                  *_bias;
    const ITensor                                  *_mm_result;
    ITensor                                        *_output;
    int32_t                                         _a_offset;
    int32_t                                         _b_offset;
    int32_t                                         _k_offset;
    bool                                            _slide_vector_sum_col;
    GEMMLowpOutputStageInfo                         _output_stage;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_NEGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H__ */
