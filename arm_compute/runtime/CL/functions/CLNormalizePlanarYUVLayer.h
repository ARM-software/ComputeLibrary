/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLNORMALIZEPLANARYUVLAYER_H
#define ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLNORMALIZEPLANARYUVLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLNormalizePlanarYUVLayerKernel
 *
 *  @note The function simulates a NormalizePlanarYUV layer.
 */
class CLNormalizePlanarYUVLayer : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F32            |F32            |
     * |F16            |F16            |
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     *
     * @param[in]  input  Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, channels].
     *                    Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Destinationfeature tensor. Data type supported: same as @p input
     * @param[in]  mean   Mean values tensor. 1 dimension with size equal to the number of input channels. Data types supported: Same as @p input
     * @param[in]  std    Standard deviation values tensor. 1 dimension with size equal to the number of input channels.
     *                    Data types supported: Same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *std);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, channels].
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output          Destinationfeature tensor. Data type supported: same as @p input
     * @param[in]  mean            Mean values tensor. 1 dimension with size equal to the number of input channels. Data types supported: Same as @p input
     * @param[in]  std             Standard deviation values tensor. 1 dimension with size equal to the number of input channels.
     *                    Data types supported: Same as @p input
     */
    void configure(const CLCompileContext &compile_context,
                   const ICLTensor        *input,
                   ICLTensor              *output,
                   const ICLTensor        *mean,
                   const ICLTensor        *std);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNormalizePlanarYUVLayer
     *
     * @param[in]  input  Source tensor info. 3 lower dimensions represent a single input with dimensions [width, height, channels].
     *                    Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Destination tensor info. Data type supported: same as @p input
     * @param[in]  mean   Mean values tensor info. 1 dimension with size equal to the number of input channels. Data types supported: Same as @p input
     * @param[in]  std    Standard deviation values tensor info. 1 dimension with size equal to the number of input channels.
     *                    Data types supported: Same as @p input
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *std);
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLNORMALIZEPLANARYUVLAYER_H
