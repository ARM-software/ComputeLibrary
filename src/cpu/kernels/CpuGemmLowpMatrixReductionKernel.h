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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_REDUCTION_KERNEL_H
#define ARM_COMPUTE_CPU_GEMMLOWP_REDUCTION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
// Forward declarations
struct GEMMLowpReductionKernelInfo;
namespace cpu
{
namespace kernels
{
/** Kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class CpuGemmLowpMatrixAReductionKernel : public ICpuKernel<CpuGemmLowpMatrixAReductionKernel>
{
public:
    /** Default constructor */
    CpuGemmLowpMatrixAReductionKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpMatrixAReductionKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  src  Input tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] dst  Output row-vector of sums of all the entries in each row of mtx_a. Data type supported: S32
     * @param[in]  info Kernel metadata:
     *                            - k            (num_mtx_a_cols) Number of matrix A columns
     *                            - is_reshaped  (is_interleaved4x4) True if the matrix A has been interleaved4x4
     *                            - scalar       Scalar value to multiply each reduced row by.
     *                            - mul_byscalar True if each reduced column must be multiplied by a scalar value.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const GEMMLowpReductionKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpMatrixAReductionKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Execution of the reduction kernel specialized on the input type
     *
     * @param[in] src    Input tensor
     * @param[in] dst    Output tensor
     * @param[in] window Execution window
     */
    template <typename T>
    void run_internal(const ITensor *src, ITensor *dst, const Window &window);

    /** Common signature for all reduction functions
     *
     * @param[in]  src    Input tensor
     * @param[out] dst    Output tensor
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    using CpuGemmLowpMatrixAReductionKernelPtr = void (CpuGemmLowpMatrixAReductionKernel::*)(const ITensor *src, ITensor *dst, const Window &window);

    CpuGemmLowpMatrixAReductionKernelPtr _func{ nullptr };
    int32_t                              _k{ 0 };
    int32_t                              _scalar{ 0 };
    bool                                 _mul_by_scalar{ false };
};

/** Kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 */
class CpuGemmLowpMatrixBReductionKernel : public ICpuKernel<CpuGemmLowpMatrixBReductionKernel>
{
public:
    /** Default constructor */
    CpuGemmLowpMatrixBReductionKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpMatrixBReductionKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  src  Input tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] dst  Output row-vector of sums of all the entries in each column of mtx_b. Data type supported: S32
     * @param[in]  info Kernel metadata:
     *                            - k            (num_mtx_b_rows) Number of matrix B rows.
     *                            - is_reshaped  (is_transposed1xW) True if the input tensor is transposed 1xW.
     *                            - scalar       Scalar value to multiply each reduced row by.
     *                            - mul_byscalar True if each reduced row must be multiplied by a scalar value.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const GEMMLowpReductionKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpMatrixBReductionKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const GEMMLowpReductionKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Execution of the reduction kernel specialized on the input type
     *
     * @param[in] src    Input tensor
     * @param[in] dst    Output tensor
     * @param[in] window Execution window
     * @param[in] info   Thread-related information
     */
    template <typename T>
    void run_internal(const ITensor *src, ITensor *dst, const Window &window, const ThreadInfo &info);

    /** Common signature for all reduction functions
     *
     * @param[in]  src    Input tensor
     * @param[out] dst    Output tensor
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    using CpuGemmLowpMatrixBReductionKernelPtr = void (CpuGemmLowpMatrixBReductionKernel::*)(const ITensor *src, ITensor *dst, const Window &window, const ThreadInfo &info);

    CpuGemmLowpMatrixBReductionKernelPtr _func{ nullptr };
    int32_t                              _k{ 0 };
    int32_t                              _scalar{ 0 };
    bool                                 _mul_by_scalar{ false };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMMLOWP_REDUCTION_KERNEL_H */
