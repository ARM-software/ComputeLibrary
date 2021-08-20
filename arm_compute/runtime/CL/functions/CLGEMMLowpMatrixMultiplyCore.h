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
#ifndef ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H
#define ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class IMemoryManager;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on OpenCL. */
class CLGEMMLowpMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyCore(const CLGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move constructor */
    CLGEMMLowpMatrixMultiplyCore(CLGEMMLowpMatrixMultiplyCore &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyCore &operator=(const CLGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move assignment operator */
    CLGEMMLowpMatrixMultiplyCore &operator=(CLGEMMLowpMatrixMultiplyCore &&) = default;
    /** Default destructor */
    ~CLGEMMLowpMatrixMultiplyCore();
    /** Initialise the kernel's inputs, output
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2     |dst            |
     * |:--------------|:------------------|:--------|:--------------|
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8             |S32      |QASYMM8        |
     * |QASYMM8        |QASYMM8            |S32      |S32            |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |S32            |
     * |QASYMM8        |QSYMM8             |S32      |S32            |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8             |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |S32            |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |S32            |
     * |QASYMM8_SIGNED |QSYMM8             |S32      |S32            |
     *
     * @note GEMMLowp:  low precision GEMM kernel. [A * B + C]
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from 8-bit quantized to int32 and add a_offset to each of them.
     *  -# Convert b values from 8-bit quantized to int32 and add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *  -# Quantize to uint8 if gemm_info.gemmlowp_output_stage != NONE
     *
     * @param[in]  a         First input tensor  (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] output    Output tensor. Data type supported: S32 or QASYMM8/QASYMM8_SIGNED if gemm_info.gemmlowp_output_stage != NONE
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Initialise the kernel's inputs, output
     *
     * @note GEMMLowp:  low precision GEMM kernel. [A * B + C]
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from 8-bit quantized to int32 and add a_offset to each of them.
     *  -# Convert b values from 8-bit quantized to int32 and add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *  -# Quantize to uint8 if gemm_info.gemmlowp_output_stage != NONE
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  a               First input tensor  (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  b               Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in]  c               Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] output          Output tensor. Data type supported: S32 or QASYMM8/QASYMM8_SIGNED if gemm_info.gemmlowp_output_stage != NONE
     * @param[in]  gemm_info       (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyCore
     *
     * @param[in] a         First input tensor info (Matrix A). Data type supported: QASYMM8.
     * @param[in] b         Second input tensor info (Matrix B). Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[in] c         Third input tensor info (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[in] output    Output tensor info. Data type supported: S32 or QASYMM8/QASYMM8_SIGNED if gemm_info.gemmlowp_output_stage != NONE
     * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                      if the reshape of matrix B should be executed only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H */