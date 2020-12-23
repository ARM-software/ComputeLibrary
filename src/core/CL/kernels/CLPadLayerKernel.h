/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLPADLAYERKERNEL_H
#define ARM_COMPUTE_CLPADLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the PadLayer function. */
class CLPadLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLPadLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPadLayerKernel(const CLPadLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPadLayerKernel &operator=(const CLPadLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLPadLayerKernel(CLPadLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLPadLayerKernel &operator=(CLPadLayerKernel &&) = default;
    /** Default destructor */
    ~CLPadLayerKernel() = default;
    /** Set the input and output tensor.
     *
     * @param[in]  input          Source tensor. Data types supported: All.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value (Optional) Constant value to be used for the padding.
     * @param[in]  mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    void configure(const ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);
    /** Set the input and output tensor.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: All.
     * @param[out] output          Output tensor. Data type supported: same as @p input
     * @param[in]  padding         The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                             specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value  (Optional) Constant value to be used for the padding.
     * @param[in]  mode            (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                             or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value = PixelValue(),
                   PaddingMode mode = PaddingMode::CONSTANT);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPadLayerKernel
     *
     * @param[in] input          Source tensor info. Data types supported: All.
     * @param[in] output         Output tensor info. Data type supported: same as @p input
     * @param[in] padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                           specifies the front and the end padding in the i-th dimension.
     * @param[in] constant_value (Optional) Constant value to be used for the padding.
     * @param[in] mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT,
     *                            or reflect the input, either including the border values (SYMMETRIC) or not (REFLECT).
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value = PixelValue(), PaddingMode mode = PaddingMode::CONSTANT);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    bool             _4d_enabled;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLPADLAYERKERNEL_H */
