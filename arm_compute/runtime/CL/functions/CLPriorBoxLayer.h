/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLPRIORBOXLAYER_H
#define ARM_COMPUTE_CLPRIORBOXLAYER_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class CLPriorBoxLayerKernel;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLPriorBoxLayerKernel. */
class CLPriorBoxLayer : public ICLSimpleFunction
{
public:
    /** Constructor */
    CLPriorBoxLayer();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0     |src1     |dst      |
     * |:--------|:--------|:--------|
     * |F32      |F32      |F32      |
     *
     * @param[in]  input1 First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2 Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data types and layouts supported: same as @p input1
     * @param[in]  info   Prior box layer info.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2          Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output          Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data types and layouts supported: same as @p input1
     * @param[in]  info            Prior box layer info.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPriorBoxLayer
     *
     * @param[in] input1 First source tensor info. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in] input2 Second source tensor info. Data types and layouts supported: same as @p input1
     * @param[in] output Destination tensor info. Output dimensions are [W * H * num_priors * 4, 2]. Data types and layouts supported: same as @p input1
     * @param[in] info   Prior box layer info.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info);

private:
    cl::Buffer _min;
    cl::Buffer _max;
    cl::Buffer _aspect_ratios;
};
} // arm_compute
#endif /* ARM_COMPUTE_CLPRIORBOXLAYER_H */
