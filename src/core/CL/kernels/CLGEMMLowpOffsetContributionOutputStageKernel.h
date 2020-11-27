/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H
#define ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel used to add the offset contribution after the matrix multiplication and perform the output stage.
 *
 * This kernel takes a final int32 accumulator value (the output of the matrix multiplication), adds to it the offset contribution
 * of matrix A and matrix B and performs the output stage defined by the output_stage argument
 *
 * @note For quantized computations the output data type for auto-initialization must be passed as part of the @ref GEMMLowpOutputStageInfo.
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
     * @param[in]  mm_result          Input tensor containing the result of the matrix multiplication. Data type supported: S32
     * @param[in]  vector_sum_col     Input row-vector of sums of all the entries in each column of matrix B.
     *                                Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  vector_sum_row     Input row-vector of sums of all the entries in each row of matrix A.
     *                                Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  bias               Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                                Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output             Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  k                  Number of matrix A columns or Matrix B rows
     * @param[in]  a_offset           Offset to be added to each element of the matrix A.
     * @param[in]  b_offset           Offset to be added to each element of the matrix B.
     * @param[in]  output_stage       GEMMLowp output stage info
     * @param[in]  output_multipliers Output multipliers tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32
     * @param[in]  output_shifts      Output shifts tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32
     */
    void configure(const ICLTensor *mm_result, const ICLTensor *vector_sum_col, const ICLTensor *vector_sum_row, const ICLTensor *bias, ICLTensor *output, int32_t k, int32_t a_offset, int32_t b_offset,
                   const GEMMLowpOutputStageInfo &output_stage, const ICLTensor *output_multipliers, const ICLTensor *output_shifts);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context    The compile context to be used.
     * @param[in]  mm_result          Input tensor containing the result of the matrix multiplication. Data type supported: S32
     * @param[in]  vector_sum_col     Input row-vector of sums of all the entries in each column of matrix B.
     *                                Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  vector_sum_row     Input row-vector of sums of all the entries in each row of matrix A.
     *                                Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in]  bias               Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                                Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output             Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  k                  Number of matrix A columns or Matrix B rows
     * @param[in]  a_offset           Offset to be added to each element of the matrix A.
     * @param[in]  b_offset           Offset to be added to each element of the matrix B.
     * @param[in]  output_stage       GEMMLowp output stage info
     * @param[in]  output_multipliers Output multipliers tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32
     * @param[in]  output_shifts      Output shifts tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *mm_result, const ICLTensor *vector_sum_col, const ICLTensor *vector_sum_row, const ICLTensor *bias, ICLTensor *output,
                   int32_t k,
                   int32_t a_offset, int32_t b_offset,
                   const GEMMLowpOutputStageInfo &output_stage, const ICLTensor *output_multipliers, const ICLTensor *output_shifts);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpOffsetContributionKernel
     *
     * @param[in] mm_result          Input tensor containing the result of @ref CLGEMMLowpOffsetContributionKernel. Data type supported: S32
     * @param[in] vector_sum_col     Input row-vector of sums of all the entries in each column of matrix B.
     *                               Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in] vector_sum_row     Input row-vector of sums of all the entries in each row of matrix A.
     *                               Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in] bias               Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                               Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output             Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in] a_offset           Offset to be added to each element of the matrix A.
     * @param[in] b_offset           Offset to be added to each element of the matrix B.
     * @param[in] output_stage       GEMMLowp output stage info
     * @param[in] output_multipliers Output multipliers tensor info. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                               Supported data types: S32
     * @param[in] output_shifts      Output shifts tensor info. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                               Supported data types: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, const ITensorInfo *output, int32_t a_offset,
                           int32_t b_offset, const GEMMLowpOutputStageInfo &output_stage, const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_mm_result;
    const ICLTensor *_vector_sum_col;
    const ICLTensor *_vector_sum_row;
    const ICLTensor *_bias;
    ICLTensor       *_output;
    const ICLTensor *_output_multipliers;
    const ICLTensor *_output_shifts;
    bool             _is_quantized_per_channel;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLGEMMLOWPOFFSETCONTRIBUTIONOUTPUTSTAGEKERNEL_H */
