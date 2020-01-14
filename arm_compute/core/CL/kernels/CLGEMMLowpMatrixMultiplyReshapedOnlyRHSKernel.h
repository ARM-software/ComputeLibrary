/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H
#define ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply matrices with QASYMM8 data type when only the input matrix RHS (input1) has been reshaped
 *
 * @note The input matrix input1 must be reshaped through @ref CLGEMMReshapeRHSMatrixKernel
 * @note For fused output stage, only GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT type is supported
 */
class CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel : public ICLKernel
{
public:
    /** Default Constructor */
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel(const CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &operator=(const CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &operator=(CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input0             Input tensor containing the LHS matrix. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  input1             Input tensor containing the RHS reshaped matrix. Data type supported: same as @p input0
     * @param[out] output             Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/S32.
     * @param[in]  gemm_info          GEMM information used to retrieve the original dimensions of the input matrices, output stage information and RHS/LHS info.
     *                                Only the following values are supported for LHS info:
     *                                lhs_info.m0: 2,3,4,5,6,7,8
     *                                lhs_info.k0: 2,3,4,8,16
     *                                Only the following values are supported for RHS info:
     *                                rhs_info.n0: 2,3,4,8,16
     *                                rhs_info.k0: same as lhs_info.k0
     *                                rhs_info.transpose: true
     * @param[in]  vector_sum_col     (Optional) Input row-vector of sums of all the entries in each column of matrix B.
     *                                Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: S32
     * @param[in]  vector_sum_row     (Optional) Input row-vector of sums of all the entries in each row of matrix A.
     *                                Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: S32
     * @param[in]  bias               (Optional) Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                                Biases are 1D tensor with dimensions [OFM]. Data type supported: S32.
     * @param[in]  output_multipliers (Optional) Output multipliers tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32.
     * @param[in]  output_shifts      (Optional) Output shifts tensor. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                                Supported data types: S32.
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, const GEMMKernelInfo &gemm_info, const ICLTensor *vector_sum_col = nullptr,
                   const ICLTensor *vector_sum_row = nullptr, const ICLTensor *bias = nullptr, const ICLTensor *output_multipliers = nullptr, const ICLTensor *output_shifts = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel
     *
     * @param[in] input0             Input tensor info for the LHS matrix. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in] input1             Input tensor info for the RHS reshaped matrix. Data type supported: same as @p input0
     * @param[in] output             Output tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/S32.
     * @param[in] gemm_info          GEMM information used to retrieve the original dimensions of the input matrices, output stage information and RHS/LHS info.
     *                               Only the following values are supported for LHS info:
     *                               lhs_info.m0: 2,3,4,5,6,7,8
     *                               lhs_info.k0: 2,3,4,8,16
     *                               Only the following values are supported for RHS info:
     *                               rhs_info.n0: 2,3,4,8,16
     *                               rhs_info.k0: same as lhs_info.k0
     *                               rhs_info.transpose: true
     * @param[in] vector_sum_col     (Optional) Input row-vector info of sums of all the entries in each column of matrix B.
     *                               Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: S32
     * @param[in] vector_sum_row     (Optional) Input row-vector info of sums of all the entries in each row of matrix A.
     *                               Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: S32
     * @param[in] bias               (Optional) Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                               Biases are 1D tensor with dimensions [OFM]. Data type supported: S32.
     * @param[in] output_multipliers (Optional) Output multipliers tensor info. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                               Supported data types: S32.
     * @param[in] output_shifts      (Optional) Output shifts tensor info. In case of per-channel quantization, the number of multipliers must be equal to the number of filters (OFM).
     *                               Supported data types: S32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, const GEMMKernelInfo &gemm_info, const ITensorInfo *vector_sum_col = nullptr,
                           const ITensorInfo *vector_sum_row = nullptr, const ITensorInfo *bias = nullptr, const ITensorInfo *output_multipliers = nullptr,
                           const ITensorInfo *output_shifts = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    ICLTensor       *_output;
    const ICLTensor *_vector_sum_col;
    const ICLTensor *_vector_sum_row;
    const ICLTensor *_bias;
    const ICLTensor *_output_multipliers;
    const ICLTensor *_output_shifts;
    bool             _slide_matrix_b;
    bool             _reinterpret_input_as_3d;
    bool             _reinterpret_output_as_3d;
    bool             _use_dummy_work_items;
    bool             _is_quantized_per_channel;
    bool             _fuse_output_stage;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H */