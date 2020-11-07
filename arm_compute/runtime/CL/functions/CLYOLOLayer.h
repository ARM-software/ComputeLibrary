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
#ifndef ARM_COMPUTE_CLYOLOLAYER_H
#define ARM_COMPUTE_CLYOLOLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLYOLOLayerKernel that performs a partial activation on the input
 *
 *  For each box, activate only:
 *    - x and y position (channel 0 and 1 of each box)
 *    - objectiveness    (channel 4 of each box)
 *    - classes          (channel 5 to (classes - 5) of each box)
 */
class CLYOLOLayer : public ICLSimpleFunction
{
public:
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in, out] input       Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                             of the activation function. Data types supported: F16/F32.
     * @param[out]     output      Destination tensor. Data type supported: same as @p input
     * @param[in]      act_info    Activation layer parameters.
     * @param[in]      num_classes Number of classes to activate (must be submultiple of @p input channels)
     */
    void configure(ICLTensor *input, ICLTensor *output, const ActivationLayerInfo &act_info, int32_t num_classes);
    /** Set the input and output tensor.
     *
     * @note If the output tensor is a nullptr or is equal to the input, the activation function will be performed in-place
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input           Source tensor. In case of @p output tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: F16/F32.
     * @param[out]     output          Destination tensor. Data type supported: same as @p input
     * @param[in]      act_info        Activation layer parameters.
     * @param[in]      num_classes     Number of classes to activate (must be submultiple of @p input channels)
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const ActivationLayerInfo &act_info, int32_t num_classes);
    /** Static function to check if given info will lead to a valid configuration of @ref CLYOLOLayer
     *
     * @param[in] input       Source tensor info. In case of @p output tensor info = nullptr, this tensor will store the result
     *                        of the activation function. Data types supported: F16/F32.
     * @param[in] output      Destination tensor info. Data type supported: same as @p input
     * @param[in] act_info    Activation layer information.
     * @param[in] num_classes Number of classes to activate (must be submultiple of @p input channels)
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLYOLOLAYER_H */
