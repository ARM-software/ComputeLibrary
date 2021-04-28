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
#ifndef ARM_COMPUTE_CLNORMALIZATIONLAYER_H
#define ARM_COMPUTE_CLNORMALIZATIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class CLFillBorderKernel;
class CLNormalizationLayerKernel;
class ICLTensor;
class ITensorInfo;

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
    /** Prevent instances of this class from being copied */
    CLNormalizationLayer(const CLNormalizationLayer &) = delete;
    /** Prevent instances of this class from being copied */
    CLNormalizationLayer &operator=(const CLNormalizationLayer &) = delete;
    /** Prevent instances of this class to be moved */
    CLNormalizationLayer(CLNormalizationLayer &&) = delete;
    /** Prevent instances of this class to be moved */
    CLNormalizationLayer &operator=(CLNormalizationLayer &&) = delete;
    /** Default destructor */
    ~CLNormalizationLayer();
    /** Set the input and output tensors.
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
     * @param[in, out] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           and an optional 4th dimension for batch of inputs. Data types supported: F16/F32 (Written to by the border handler).
     *                           Data layouts supported: NCHW/NHWC.
     * @param[out]     output    Destination tensor. Dimensions, data type and number of channels must match the input ones.
     *                           Data types supported: same as @p input. Data layouts supported: same as @p input.
     * @param[in]      norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(ICLTensor *input, ICLTensor *output, const NormalizationLayerInfo &norm_info);
    /** Set the input and output tensors.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                                 and an optional 4th dimension for batch of inputs. Data types supported: F16/F32 (Written to by the border handler).
     *                                 Data layouts supported: NCHW/NHWC.
     * @param[out]     output          Destination tensor. Dimensions, data type and number of channels must match the input ones.
     *                                 Data types supported: same as @p input. Data layouts supported: same as @p input.
     * @param[in]      norm_info       Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const NormalizationLayerInfo &norm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNormalizationLayer
     *
     * @param[in] input     Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                      and an optional 4th dimension for batch of inputs. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in] output    Destination tensor. Dimensions, data type and number of channels must match the input ones.
     *                      Data layouts supported: same as @p input.
     * @param[in] norm_info Normalization layer information like the normalization type, normalization size and other parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const NormalizationLayerInfo &norm_info);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLNormalizationLayerKernel> _norm_kernel;    /**< Normalization layer kernel to run */
    std::unique_ptr<CLFillBorderKernel>         _border_handler; /**< Kernel to handle  borders */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLNORMALIZATIONLAYER_H */
