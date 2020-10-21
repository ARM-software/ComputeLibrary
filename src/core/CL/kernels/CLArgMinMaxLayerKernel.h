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
#ifndef ARM_COMPUTE_CLARGMINMAXLAYERKERNEL_H
#define ARM_COMPUTE_CLARGMINMAXLAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reduction operation kernel
 *
 * @note The default data type for an uninitialized output tensor is
 *       signed 32-bit integer (S32). It is the user's responsibility to check
 *       that the results do not overflow because the indices are computed
 *       in unsigned 32-bit (U32).
 */
class CLArgMinMaxLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLArgMinMaxLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArgMinMaxLayerKernel(const CLArgMinMaxLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArgMinMaxLayerKernel &operator=(const CLArgMinMaxLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLArgMinMaxLayerKernel(CLArgMinMaxLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLArgMinMaxLayerKernel &operator=(CLArgMinMaxLayerKernel &&) = default;
    /** Default destructor */
    ~CLArgMinMaxLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input       Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  prev_output Destination tensor of the previous iterations of @ref CLArgMinMaxLayerKernel. Data types supported: U32/S32
     *                         Has to be nullptr for the first iteration
     * @param[out] output      Destination tensor. Data types supported: U32/S32
     *                         Output will have the same number of dimensions as input.
     * @param[in]  axis        Axis along which to reduce. Supported reduction axis : 0,1,2,3
     * @param[in]  op          Reduction operation to perform. Only ArgMin and ArgMax are supported.
     */
    void configure(const ICLTensor *input, const ICLTensor *prev_output, ICLTensor *output, unsigned int axis, ReductionOperation op);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in]  prev_output     Destination tensor of the previous iterations of @ref CLArgMinMaxLayerKernel. Data types supported: U32/S32
     *                             Has to be nullptr for the first iteration
     * @param[out] output          Destination tensor. Data types supported: U32/S32
     *                             Output will have the same number of dimensions as input.
     * @param[in]  axis            Axis along which to reduce. Supported reduction axis : 0,1,2,3
     * @param[in]  op              Reduction operation to perform. Only ArgMin and ArgMax are supported.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *prev_output, ICLTensor *output, unsigned int axis, ReductionOperation op);

    /** Static function to check if given info will lead to a valid configuration of @ref CLArgMinMaxLayerKernel.
     *
     * @param[in] input       Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/S32/F16/F32.
     * @param[in] prev_output Destination tensor info of the previous iterations. Data types supported: U32/S32
     *                        Has to be nullptr for the first iteration
     * @param[in] output      Destination tensor info. Data types supported: U32/S32
     *                        Output will have the same number of dimensions as input.
     * @param[in] axis        Axis along which to reduce. Supported reduction axis : 0,1,2,3
     * @param[in] op          Reduction operation to perform.  Only ArgMin and ArgMax are supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *prev_output, const ITensorInfo *output, unsigned int axis, ReductionOperation op);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor   *_input;
    const ICLTensor   *_prev_output;
    ICLTensor         *_output;
    unsigned int       _reduction_axis;
    ReductionOperation _op;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLARGMINMAXLAYERKERNEL_H */
