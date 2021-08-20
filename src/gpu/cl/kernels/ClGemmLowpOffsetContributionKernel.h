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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL_H
#define ARM_COMPUTE_CL_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel used to add the offset contribution after the matrix multiplication. The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of the matrix multiplication),
 * and adds to it the offset contribution of matrix A and matrix B in-place.
 *
 * The final result is:
 *
 * mm_result[i][k] = mm_result[i][k] +
 *                   (vector_sum_col[k] * a_offset) +
 *                   (vector_sum_row[i] * b_offset) +
 *                   (a_offset * b_offset * k)
 *
 */
class ClGemmLowpOffsetContributionKernel : public IClKernel
{
public:
    ClGemmLowpOffsetContributionKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmLowpOffsetContributionKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] mm_result       Input tensor containing the result of the matrix multiplication. Data type supported: S32
     * @param[in]      vector_sum_col  Input row-vector of sums of all the entries in each column of matrix B.
     *                                 Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: same as @p mm_result
     * @param[in]      vector_sum_row  Input row-vector of sums of all the entries in each row of matrix A.
     *                                 Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: same as @p mm_result
     * @param[in]      bias            Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                                 Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in]      k               Number of matrix A columns or Matrix B rows
     * @param[in]      a_offset        Offset to be added to each element of the matrix A.
     * @param[in]      b_offset        Offset to be added to each element of the matrix B.
     */
    void configure(const CLCompileContext &compile_context,
                   const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                   int32_t k, int32_t a_offset, int32_t b_offset);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmLowpOffsetContributionKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mm_result, const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias, int32_t a_offset, int32_t b_offset);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_OFFSET_CONTRIBUTION_KERNEL_H */
