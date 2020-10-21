/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLL2NORMALIZELAYERKERNEL_H
#define ARM_COMPUTE_CLL2NORMALIZELAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for performing a L2 normalize on a given axis given the square sum of it in this axis */
class CLL2NormalizeLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLL2NormalizeLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLL2NormalizeLayerKernel(const CLL2NormalizeLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLL2NormalizeLayerKernel &operator=(const CLL2NormalizeLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLL2NormalizeLayerKernel(CLL2NormalizeLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLL2NormalizeLayerKernel &operator=(CLL2NormalizeLayerKernel &&) = default;
    /** Default destructor */
    ~CLL2NormalizeLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  sum     Sum values tensor. Data types supported: same as @p input.
     *                     Sum will have the same number of dimensions as input.
     * @param[out] output  Destination tensor. Data types and data layouts supported: Same as @p input.
     *                     Output will have the same number of dimensions as input.
     * @param[in]  axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in]  epsilon Lower bound value for the normalization.
     */
    void configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, int axis, float epsilon);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  sum             Sum values tensor. Data types supported: same as @p input.
     *                             Sum will have the same number of dimensions as input.
     * @param[out] output          Destination tensor. Data types and data layouts supported: Same as @p input.
     *                             Output will have the same number of dimensions as input.
     * @param[in]  axis            Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in]  epsilon         Lower bound value for the normalization.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, int axis, float epsilon);

    /** Static function to check if given info will lead to a valid configuration of @ref CLL2NormalizeLayerKernel.
     *
     * @param[in] input   Source tensor info. Data types supported: F16/F32. Data layouts supported: NCHW/NHWC.
     * @param[in] sum     Sum values tensor info. Data types supported: same as @p input.
     *                    Sum will have the same number of dimensions as input.
     * @param[in] output  Destination tensor info. Data types and data layouts supported: Same as @p input.
     *                    Output will have the same number of dimensions as input.
     * @param[in] axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in] epsilon Lower bound value for the normalization.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_sum;
    ICLTensor       *_output;
    unsigned int     _actual_axis;
    float            _epsilon;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLL2NORMALIZELAYERKERNEL_H */
