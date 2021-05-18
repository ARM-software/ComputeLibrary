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
#ifndef ARM_COMPUTE_CLPADLAYER_H
#define ARM_COMPUTE_CLPADLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLCopy.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class CLCompileContext;
class CLPadLayerKernel;
class ICLTensor;

/** Basic function to pad a tensor. This function calls the following OpenCL functions/kernels:
 *
 *  -# @ref CLPadLayerKernel if there is padding to be added
 *  -# @ref CLCopy otherwise
 */
class CLPadLayer : public IFunction
{
public:
    /** Default constructor */
    CLPadLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPadLayer(const CLPadLayer &) = delete;
    /** Default move constructor */
    CLPadLayer(CLPadLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPadLayer &operator=(const CLPadLayer &) = delete;
    /** Default move assignment operator */
    CLPadLayer &operator=(CLPadLayer &&) = default;
    /** Default destructor */
    ~CLPadLayer();

    /** Initialize the function
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src      |dst       |
     * |:--------|:---------|
     * |All      |All       |
     *
     * @param[in]  input          Source tensor. Data types supported: All.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value (Optional) Constant value to be used for the padding.
     * @param[in]  mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    void configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: All.
     * @param[out] output          Output tensor. Data type supported: same as @p input
     * @param[in]  padding         The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                             specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value  (Optional) Constant value to be used for the padding.
     * @param[in]  mode            (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value = PixelValue(),
                   PaddingMode mode = PaddingMode::CONSTANT);

    /**  Static function to check if given info will lead to a valid configuration of @ref CLPadLayer.
     *
     * @param[in] input          Source tensor info. Data types supported: All.
     * @param[in] output         Output tensor info. Data type supported: same as @p input
     * @param[in] padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                           specifies the front and the end padding in the i-th dimension.
     * @param[in] constant_value (Optional) Constant value to be used for the padding
     * @param[in] mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);

    // Inherited methods overridden:
    void run() override;

private:
    void configure_reflect_mode(ICLTensor *input, ICLTensor *output);

    std::unique_ptr<CLPadLayerKernel> _pad_kernel;
    CLCopy                            _copy;
    bool                              _perform_pad;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_PADLAYER_H */
