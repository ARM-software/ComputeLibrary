/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_OFFSETCONTRIBUTION_OUTPUTSTAGE_KERNEL_H
#define ARM_COMPUTE_CPU_GEMMLOWP_OFFSETCONTRIBUTION_OUTPUTSTAGE_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel used to add the offset contribution and perform the output stage after @ref CpuGemmLowpMatrixMultiplyKernel.
 *
 * The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel),
 * and adds to it the offset contribution of matrix A and matrix B in-place.
 *
 * The output stage can perform either QuantizeDownInt32ToUint8Scale or QuantizeDownInt32ToUint8ScaleByFixedPoint for Uint8.
 * The output stage can perform either QuantizeDownInt32ToInt8Scale or QuantizeDownInt32ToInt8ScaleByFixedPoint for Int8.
 *
 * For QuantizeDownInt32ToUint8Scale/QuantizeDownInt32ToInt8Scale the final result is:
 *
 * ((mm_result'[i][k] + result_offset) * result_mult_int) >> result_shift
 *
 * For QuantizeDownInt32ToUint8ScaleByFixedPoint/QuantizeDownInt32ToInt8ScaleByFixedPoint the final result is:
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

class CpuGemmLowpOffsetContributionOutputStageKernel : public ICpuKernel<CpuGemmLowpOffsetContributionOutputStageKernel>
{
public:
    /** Default constructor */
    CpuGemmLowpOffsetContributionOutputStageKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpOffsetContributionOutputStageKernel);
    /** Initialise the kernel inputs and output.
     *
     * @param[in]  mm_result      Input tensor info containing the result of @ref CpuGemmLowpMatrixMultiplyKernel. Data type supported: S32
     * @param[in]  vector_sum_col Input row-vector tensor info of sums of all the entries in each column of matrix B.
     *                            Can be a 1D or 2D tensor, in case of 2D, y dim is the batch dimension
     *                            Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  vector_sum_row Input row-vector tensor info of sums of all the entries in each row of matrix A.
     *                            Can be a 1D or 2D tensor, in case of 2D, y dim is the batch dimension
     * @param[in]  bias           Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                            Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p mm_result.
     * @param[out] dst            Output tensor info containing the final quantized result. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  k              Number of matrix A columns or Matrix B rows
     * @param[in]  a_offset       Offset to be added to each element of the matrix A.
     * @param[in]  b_offset       Offset to be added to each element of the matrix B.
     * @param[in]  output_stage   GEMMLowp output stage info, providing the type of quantization and the necessary parameters.
     */
    void configure(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, ITensorInfo *dst, int32_t k, int32_t a_offset,
                   int32_t                 b_offset,
                   GEMMLowpOutputStageInfo output_stage);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpOffsetContributionOutputStageKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *dst, int32_t a_offset,
                           int32_t                 b_offset,
                           GEMMLowpOutputStageInfo output_stage);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Function to use for the particular tensors passed to configure() */
    int32_t                 _a_offset{ 0 };
    int32_t                 _b_offset{ 0 };
    int32_t                 _k_offset{ 0 };
    bool                    _is_vector_sum_col_batched{ true };
    GEMMLowpOutputStageInfo _output_stage{ GEMMLowpOutputStageInfo() };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMMLOWP_OFFSETCONTRIBUTION_OUTPUTSTAGE_KERNEL_H */
