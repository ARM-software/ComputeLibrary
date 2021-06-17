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
#ifndef ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H
#define ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

/** This file contains all available output stages for GEMMLowp.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class ITensor;
class ITensorInfo;
/** Basic function to execute GEMMLowpQuantizeDown kernels.
 *
 *  This function calls the following operators:
 *
 * -# @ref cpu::CpuGemmLowpOutputStage
*/
class NEGEMMLowpOutputStage : public IFunction
{
public:
    /** Constructor */
    NEGEMMLowpOutputStage();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpOutputStage(const NEGEMMLowpOutputStage &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpOutputStage &operator=(const NEGEMMLowpOutputStage &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpOutputStage(NEGEMMLowpOutputStage &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpOutputStage &operator=(NEGEMMLowpOutputStage &&) = delete;
    /** Default destructor */
    ~NEGEMMLowpOutputStage();
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
     * @param[in]  input  Input tensor. Data type supported: S32
     * @param[in]  bias   Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                    Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output Output tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in]  info   GEMMLowp output stage metadata.
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpOutputStage
     *
     * @param[in] input  Input tensor info. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in] info   GEMMLowp output stage metadata.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H */
