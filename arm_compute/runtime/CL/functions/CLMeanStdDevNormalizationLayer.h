/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLMEANSTDDEVNORMALIZATIONLAYER_H
#define ARM_COMPUTE_CLMEANSTDDEVNORMALIZATIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute mean and standard deviation normalization by calling @ref CLMeanStdDevNormalizationKernel */
class CLMeanStdDevNormalizationLayer : public ICLSimpleFunction
{
public:
    /** Initialise the function's input and outputs.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |F32      |F32       |
     * |F16      |F16       |
     *
     * @note If the output tensor is a nullptr, the normalization will be performed in-place.
     *
     * @param[in, out] input   Input tensor with 2 dimensions. Data types supported: F16/F32.
     * @param[out]     output  (Optional) Destination tensor. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in]      epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     */
    void configure(ICLTensor *input, ICLTensor *output = nullptr, float epsilon = 1e-8f);
    /** Initialise the function's input and outputs.
     *
     * @note If the output tensor is a nullptr, the normalization will be performed in-place.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Input tensor with 2 dimensions. Data types supported: F16/F32.
     * @param[out]     output          (Optional) Destination tensor. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in]      epsilon         (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output = nullptr, float epsilon = 1e-8f);
    /** Static function to check if given info will lead to a valid configuration of @ref CLMeanStdDevNormalizationKernel
     *
     * @param[in] input   Source tensor info with 2 dimensions. In case of @p output tensor info = nullptr,
     *                    this tensor will store the result of the normalization. Data types supported: F16/F32.
     * @param[in] output  (Optional) Destination tensor info. It can be nullptr in case of in-place computation. Data type supported: same as @p input
     * @param[in] epsilon (Optional) Small float to avoid division by zero in case of zero standard deviation. Defaults to 1e-8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output = nullptr, float epsilon = 1e-8f);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLMEANSTDDEVNORMALIZATIONLAYER_H */
