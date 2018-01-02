/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLNORMALIZATIONLAYER_H__
#define __ARM_COMPUTE_CLNORMALIZATIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLNormalizationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to compute a normalization layer. This function calls the following CL kernels:
 *
 * -# @ref CLFillBorderKernel
 * -# @ref CLNormalizationLayerKernel
 *
 */
class CLNormalizationLayer : public IFunction
{
public:
    /** Default constructor */
    CLNormalizationLayer();
    /** Set the input and output tensors.
     *
     * @param[in, out] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           and an optional 4th dimension for batch of inputs. Data types supported: QS8/QS16/F16/F32 (Written to by the border handler)
     * @param[out]     output    Destination tensor. Dimensions, data type and number of channels must match the input ones.
     * @param[in]      norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(ICLTensor *input, ICLTensor *output, const NormalizationLayerInfo &norm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNormalizationLayer
     *
     * @param[in] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                      and an optional 4th dimension for batch of inputs. Data types supported: QS8/QS16/F16/F32
     * @param[in] output    Destination tensor. Dimensions, data type and number of channels must match the input ones.
     * @param[in] norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const NormalizationLayerInfo &norm_info);

    // Inherited methods overridden:
    void run() override;

private:
    CLNormalizationLayerKernel _norm_kernel;    /**< Normalization layer kernel to run */
    CLFillBorderKernel         _border_handler; /**< Kernel to handle  borders */
};
}
#endif /* __ARM_COMPUTE_CLNORMALIZATIONLAYER_H__ */
