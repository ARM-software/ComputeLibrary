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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_OUTPUT_STAGE_H
#define ARM_COMPUTE_CL_GEMMLOWP_OUTPUT_STAGE_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

/** This file contains all available output stages for GEMMLowp on OpenCL.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
namespace opencl
{
/** Basic function to execute GEMMLowpQuantizeDown kernels on CL.
 *
 *  This function calls the following CL kernels:
 *
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleKernel
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFloatKernel
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel
*/
class ClGemmLowpOutputStage : public IClOperator
{
public:
    /** Constructor */
    ClGemmLowpOutputStage() = default;
    /** Initialise the kernel's inputs, output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1          |dst           |
     * |:--------------|:-------------|:-------------|
     * |S32            |S32           |QASYMM8       |
     * |S32            |S32           |QASYMM8_SIGNED|
     * |S32            |S32           |QSYMM16       |
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p src.
     * @param[out] dst             Destination tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  info            GEMMLowp output stage metadata.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *bias, ITensorInfo *dst, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClGemmLowpOutputStage::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo &info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_OUTPUT_STAGE_H */
