/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLPRIORBOXLAYERKERNEL_H
#define ARM_COMPUTE_CLPRIORBOXLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the PriorBox layer kernel. */
class CLPriorBoxLayerKernel : public ICLKernel
{
public:
    /** Constructor */
    CLPriorBoxLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPriorBoxLayerKernel(const CLPriorBoxLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPriorBoxLayerKernel &operator=(const CLPriorBoxLayerKernel &) = delete;
    /** Default Move Constructor. */
    CLPriorBoxLayerKernel(CLPriorBoxLayerKernel &&) = default;
    /** Default move assignment operator */
    CLPriorBoxLayerKernel &operator=(CLPriorBoxLayerKernel &&) = default;
    /** Default destructor */
    ~CLPriorBoxLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input1        First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2        Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output        Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data types and layouts supported: same as @p input1
     * @param[in]  info          Prior box layer info.
     * @param[in]  min           Minimum prior box values
     * @param[in]  max           Maximum prior box values
     * @param[in]  aspect_ratios Aspect ratio values
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info, cl::Buffer *min, cl::Buffer *max, cl::Buffer *aspect_ratios);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2          Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output          Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data types and layouts supported: same as @p input1
     * @param[in]  info            Prior box layer info.
     * @param[in]  min             Minimum prior box values
     * @param[in]  max             Maximum prior box values
     * @param[in]  aspect_ratios   Aspect ratio values
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info, cl::Buffer *min, cl::Buffer *max,
                   cl::Buffer *aspect_ratios);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPriorBoxLayerKernel
     *
     * @param[in] input1 First source tensor info. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in] input2 Second source tensor info. Data types and layouts supported: same as @p input1
     * @param[in] output Destination tensor info. Output dimensions are [W * H * num_priors * 4, 2]. Data type supported: same as @p input1
     * @param[in] info   Prior box layer info.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1;
    const ICLTensor *_input2;
    ICLTensor        *_output;
    PriorBoxLayerInfo _info;
    int               _num_priors;
    cl::Buffer       *_min;
    cl::Buffer       *_max;
    cl::Buffer       *_aspect_ratios;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLPRIORBOXLAYERKERNEL_H */
