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
#ifndef ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLMATMUL
#define ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLMATMUL

#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
// Forward declarations for used types instead of including their header, that could minimize compile time
class CLCompileContext;
class ICLTensor;
class ITensorInfo;
class MatMulInfo;
class Status;

/** Settings for MatMul OpenCL implementation */
class GpuMatMulSettings
{
public:
    /* Placeholder for operator parity between CPU/GPU */
};

/** Basic function to execute MatMul (Matrix Multiplication) on OpenCL */
class CLMatMul : public IFunction
{
public:
    /** Default constructor.*/
    CLMatMul();
    /** Default destructor */
    ~CLMatMul();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMatMul(const CLMatMul &) = delete;
    /** Default move constructor */
    CLMatMul(CLMatMul &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMatMul &operator=(const CLMatMul &) = delete;
    /** Default move assignment operator */
    CLMatMul &operator=(CLMatMul &&);
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
     * @param[in]  lhs             Left-hand side tensor info containing the input activations as Matrix A. Data types supported: F16/F32/QASYMM8_SIGNED/QASYMM8.
     * @param[in]  rhs             Right-hand side tensor info containing the input weights as Matrix B. Data types supported: same as @p lhs.
     * @param[out] dst             Output tensor to store the result of the batched matrix multiplication. Data types supported: same as @p lhs.
     * @param[in]  matmul_info     Contains MatMul operation information described in @ref MatMulInfo.
     * @param[in]  settings        Class containing flags for function level settings
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *rhs, ICLTensor *lhs, ICLTensor *dst, const MatMulInfo &matmul_info, const GpuMatMulSettings &settings = GpuMatMulSettings{});
    /** Initialise the kernel's inputs and output
     *
     * Similar to @ref CLMatMul::configure()
     */
    void configure(ICLTensor *lhs, ICLTensor *rhs, ICLTensor *dst, const MatMulInfo &matmul_info, const GpuMatMulSettings &settings = GpuMatMulSettings{});
    /** Static function to check if given info will lead to a valid configuration of @ref CLMatMul.
     *
     * Similar to @ref CLMatMul::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *output, const MatMulInfo &matmul_info);
    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLMATMUL */
