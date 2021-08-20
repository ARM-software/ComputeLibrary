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
#ifndef ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_KERNEL_H
#define ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/core/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to multiply two input matrices "A" and "B" and add a martix "C" if provided. All elements of the output matrix will be multiplied by alpha. In case matrix C is passed, it will be added to the previous result.
 *  For the matrix C, the broadcast addition is supported if the flag "broadcast_bias" is set in the GEMMReshapeInfo object
 *
 * @note If the input tensors @p src0 and @p src1 have been reshaped respectively with @ref ClGemmReshapeLhsMatrixKernel" and @ref ClGemmReshapeRhsMatrixKernel,
 *       the flag @p is_interleaved_transposed must be set to true
 *
 * @attention @p src1 tensor must have at least 2 dimensions (matrix)
 */
class ClGemmMatrixMultiplyKernel : public IClKernel
{
public:
    ClGemmMatrixMultiplyKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmMatrixMultiplyKernel);
    /** Initialise the kernel's input, output and alpha
     *
     * @param[in]  compile_context           The compile context to be used.
     * @param[in]  src0                      Input tensor containing the Matrix A. Data types supported: F16/F32
     * @param[in]  src1                      Input tensor containing the Matrix B. Data type supported: same as @p src0
     * @param[in]  src2                      Input tensor containing the Matrix C (bias). Can be nullptr. Data type supported: same as @p src0
     * @param[out] dst                       Output tensor to store the result of matrix multiplication. Data type supported: same as @p src0
     * @param[in]  alpha                     Weight of the matrix product
     * @param[in]  beta                      (Optional) Weight of vector C. Default value is 0. Only beta = 1 is currently supported.
     * @param[in]  is_interleaved_transposed (Optional) True if input0 and input1 have been reshaped respectively using @ref ClGemmReshapeLhsMatrixKernel and @ref ClGemmReshapeRhsMatrixKernel
     * @param[in]  reshape_info              (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     * @param[in]  fp_mixed_precision        (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy
     * @param[in]  activation_info           (Optional) Activation to apply after the matrix multiplication
     *
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float alpha, float beta = 0.f,
                   bool is_interleaved_transposed = true, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo(), bool fp_mixed_precision = false, const ActivationLayerInfo &activation_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmMatrixMultiplyKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta,
                           bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target, bool fp_mixed_precision = false, const ActivationLayerInfo &activation_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

public:
    bool _slide_matrix_b{ true };
    bool _reinterpret_input_as_3d{ false };
    bool _reinterpret_output_as_3d{ false };
    bool _add_bias{ false };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_KERNEL_H */
