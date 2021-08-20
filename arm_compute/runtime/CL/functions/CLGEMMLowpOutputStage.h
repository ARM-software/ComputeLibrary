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
#ifndef ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H
#define ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <limits>

/** This file contains all available output stages for GEMMLowp on OpenCL.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class CLCompileContext;
class ITensor;
class ICLTensor;
class ITensorInfo;
struct GEMMLowpOutputStageInfo;

/** Basic function to execute GEMMLowpQuantizeDown kernels on CL.
 *
 *  This function calls the following CL kernels:
 *
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleKernel
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFloatKernel
 * -# @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel
*/
class CLGEMMLowpOutputStage : public IFunction
{
public:
    CLGEMMLowpOutputStage();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpOutputStage(const CLGEMMLowpOutputStage &) = delete;
    /** Default move constructor */
    CLGEMMLowpOutputStage(CLGEMMLowpOutputStage &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpOutputStage &operator=(const CLGEMMLowpOutputStage &) = delete;
    /** Default move assignment operator */
    CLGEMMLowpOutputStage &operator=(CLGEMMLowpOutputStage &&);
    /** Default destructor */
    ~CLGEMMLowpOutputStage();
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
     * @param[out] output Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in]  info   GEMMLowp output stage metadata.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info);
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output          Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  info            GEMMLowp output stage metadata.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
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
#endif /*ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H */
