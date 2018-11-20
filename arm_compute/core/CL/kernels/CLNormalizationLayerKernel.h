/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLNORMALIZATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_CLNORMALIZATIONLAYERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the normalization layer kernel.
 */
class CLNormalizationLayerKernel : public ICLKernel
{
public:
    /** Constructor */
    CLNormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLNormalizationLayerKernel(const CLNormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLNormalizationLayerKernel &operator=(const CLNormalizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    CLNormalizationLayerKernel(CLNormalizationLayerKernel &&) = default;
    /** Default move assignment operator */
    CLNormalizationLayerKernel &operator=(CLNormalizationLayerKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                       and an optional 4th dimension for batch of inputs. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[out] output    Destination tensor. Output will have the same number of dimensions as input. Data types supported: same as @p input.
     *                       Data layouts supported: same as @p input.
     * @param[in]  norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const ICLTensor *input, ICLTensor *output, NormalizationLayerInfo norm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNormalizationLayerKernel
     *
     * @param[in] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                      and an optional 4th dimension for batch of inputs. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in] output    Destination tensor. Output will have the same number of dimensions as input. Data types supported: same as @p input.
     *                      Data layouts supported: same as @p input.
     * @param[in] norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, NormalizationLayerInfo norm_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    BorderSize       _border_size;
    bool             _is_norm_across_width;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLNORMALIZATIONLAYERKERNEL_H__ */
