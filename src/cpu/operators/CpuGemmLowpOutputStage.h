/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_OUTPUT_STAGE_H
#define ARM_COMPUTE_CPU_GEMMLOWP_OUTPUT_STAGE_H

#include "arm_compute/core/Types.h"
#include "src/cpu/ICpuOperator.h"

/** This file contains all available output stages for GEMMLowp.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
namespace cpu
{
/** Basic function to execute GEMMLowpQuantizeDown kernels.
 *
 *  This function calls the following kernels:
 *
 * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ScaleKernel
 * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
 * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel
 * -# @ref kernels::CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
*/
class CpuGemmLowpOutputStage : public ICpuOperator
{
public:
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
     * @param[in]  src  Input tensor info. Data type supported: S32
     * @param[in]  bias Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                  Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] dst  Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in]  info GEMMLowp output stage metadata.
     */
    void configure(ITensorInfo *src, ITensorInfo *bias, ITensorInfo *dst, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpOutputStage::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo &info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMMLOWP_OUTPUT_STAGE_H */
