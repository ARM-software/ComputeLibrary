/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_SRC_GPU_CL_OPERATORS_CLMATMUL
#define ACL_ARM_COMPUTE_SRC_GPU_CL_OPERATORS_CLMATMUL

#include "src/gpu/cl/IClOperator.h"
#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"
#include "src/gpu/cl/kernels/ClMatMulLowpNativeKernel.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
/** Basic operator to execute BatchMatMul on OpenCL. This operator calls the following OpenCL kernels:
 *
 *  -# @ref kernels::ClMatMulNativeKernel
 */
class ClMatMul : public IClOperator
{
public:
    /** Constructor */
    ClMatMul();
    /** Default destructor */
    ~ClMatMul() = default;
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |lhs            |rhs            |dst            |
     * |:--------------|:--------------|:--------------|
     * |F32            |F32            |F32            |
     * |F16            |F16            |F16            |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     *
     * @note BatchMatMul: Batched Matrix Multiply - [A * B], Multiplies all slices (slice is an element of a batch) of Tensors A and B
     *                    and stores the result in the dst tensor of the same batch size.
     *                    Batch here is number of slices from A and B multiplied at a time, do not confuse with the batch dimension 'N' of NHWC/NCHW
     *                    For NHWC for example: the batch is the higher dimensions H * N, and in general it is H * all higher dimensions.
     * @note All tensors must have the same data type.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  lhs             Left-hand side tensor info. Data types supported: F16/F32/QASYMM8_SIGNED/QASYMM8.
     * @param[in]  rhs             Right-hand side tensor info. Data types supported: same as @p lhs.
     * @param[out] dst             Output tensor to store the result of the batched matrix multiplication. Data types supported: same as @p lhs.
     * @param[in]  matmul_info     Contains MatMul operation information described in @ref MatMulInfo.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *dst, const MatMulInfo &matmul_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClMatMul::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulInfo &matmul_info);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<kernels::ClMatMulNativeKernel>     _matmul_native_kernel{nullptr};
    std::unique_ptr<kernels::ClMatMulLowpNativeKernel> _matmul_lowp_native_kernel{nullptr};

    bool _is_quantized{ false };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_SRC_GPU_CL_OPERATORS_CLMATMUL */
