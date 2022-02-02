/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_H
#define ARM_COMPUTE_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to multiply two input matrices "A" and "B". All elements of the output matrix/vector will be multiplied by alpha after the matrix multiplication
 *
 * @note If the output tensor is a matrix, the implementation assumes that the input tensors @p lhs and @p rhs are both matrices and reshaped respectively with @ref CpuGemmInterleave4x4Kernel" and @ref CpuGemmTranspose1xWKernel
 * @note If the output tensor is a vector and the data type is F32, the implementation assumes that the first input tensor @p lhs is a vector and the second input tensor @p rhs a matrix. The implementation also assumes that both tensors have not been reshaped
 *
 */
class CpuGemmMatrixMultiplyKernel : public ICpuKernel<CpuGemmMatrixMultiplyKernel>
{
private:
    using GemmMatrixMulKernelPtr = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const Window &, const ThreadInfo &, float, const bool)>::type;

public:
    struct GemmMatrixMulKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        GemmMatrixMulKernelPtr       ukernel;
    };

    CpuGemmMatrixMultiplyKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmMatrixMultiplyKernel);
    /** Initialise the kernel's input and output.
     *
     * @note If the output tensor is a matrix, the input matrices @p lhs and @p rhs should be the output of the kernels: @ref CpuGemmInterleave4x4Kernel and @ref CpuGemmTranspose1xWKernel
     *       These two kernels change the layout of the original matrices to be more cache-friendly.
     *
     * @param[in]  lhs            Left-handside tensor info containing the interleaved Matrix A or the vector A. Data types supported: F16/F32
     * @param[in]  rhs            Right-handside tensor info containing the transposed Matrix B if the first input tensor A is not a vector.
     *                            If the output tensor is a vector, rhs must contain the matrix B not reshaped. Data type supported: same as @p lhs
     * @param[out] dst            Output tensor to store the result of matrix multiplication. Data type supported: same as @p lhs.
     * @param[in]  alpha          Weight of the matrix product
     * @param[in]  is_interleaved (Optional) True if lhs and rhs have been reshaped respectively using @ref CpuGemmInterleave4x4Kernel and @ref CpuGemmTranspose1xWKernel
     * @param[in]  reshape_info   (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how @p lhs and @p rhs have been reshaped
     */
    void configure(const ITensorInfo *lhs, const ITensorInfo *rhs, ITensorInfo *dst, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmMatrixMultiplyKernel
     *
     * Similar to @ref CpuGemmMatrixMultiplyKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    static const std::vector<GemmMatrixMulKernel> &get_available_kernels();

private:
    /** Common signature for all the matrix multiply functions
     *
     * @param[in]  lhs    Left-handside input tensor. Data types supported: F16/F32
     * @param[in]  rhs    Right-handside input tensor. Data types supported: same as @p lhs
     * @param[out] dst    The output tensor. Data type supported: same as @p rhs
     * @param[in]  window Region on which to execute the kernel.
     * @param[in]  info   Thread info metadata.
     * @param[in]  alpha  Weight of the matrix product.
     */

    /** Matrix multiply function to use for the particular tensor types passed to configure() */
    GemmMatrixMulKernelPtr _func{ nullptr };
    float                  _alpha{ 1.f };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMM_MATRIX_MULTIPLY_KERNEL_H */
