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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_REDUCTION_KERNEL_H
#define ARM_COMPUTE_CL_GEMMLOWP_REDUCTION_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Common interface for all OpenCL reduction kernels */
class IClGemmLowpReductionKernel : public IClKernel
{
public:
    IClGemmLowpReductionKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClGemmLowpReductionKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8.
     * @param[out] output          Output row-vector of sums of all the entries in each row/col of input tensor. Data type supported: S32
     * @param[in]  info            Kernel metadata:
     *                             - k            Number of matrix columns/rows depending on the type of reduction.
     *                             - is_reshaped  True if the matrix has been reshaped.
     *                             - scalar       Scalar value to multiply each reduced column/row by.
     *                             - mul_byscalar True if each reduced column/row must be multiplied by a scalar value.
     */
    virtual void configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output, const GEMMLowpReductionKernelInfo &info) = 0;
};

/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class ClGemmLowpMatrixAReductionKernel : public IClGemmLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  mtx_a           Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8.
     * @param[out] vector_sum_row  Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in]  info            Kernel metadata:
     *                             - k            Number of matrix columns/rows depending on the type of reduction.
     *                             - is_reshaped  True if the matrix has been reshaped.
     *                             - scalar       Scalar value to multiply each reduced column/row by.
     *                             - mul_byscalar True if each reduced column/row must be multiplied by a scalar value.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *mtx_a, ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info) override;
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};

/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class ClGemmLowpMatrixBReductionKernel : public IClGemmLowpReductionKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  mtx_b           Input tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL.
     * @param[out] vector_sum_col  Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in]  info            Kernel metadata:
     *                             - k            Number of matrix columns/rows depending on the type of reduction.
     *                             - is_reshaped  True if the matrix has been reshaped.
     *                             - scalar       Scalar value to multiply each reduced column/row by.
     *                             - mul_byscalar True if each reduced column/row must be multiplied by a scalar value.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *mtx_b, ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info) override;
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_REDUCTION_KERNEL_H */
