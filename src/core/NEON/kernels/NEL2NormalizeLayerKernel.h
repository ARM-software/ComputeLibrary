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
#ifndef ARM_COMPUTE_NEL2NORMALIZELAYERKERNEL_H
#define ARM_COMPUTE_NEL2NORMALIZELAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for performing a L2 normalize on a given axis given the square sum of it in this axis */
class NEL2NormalizeLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEL2NormalizeLayerKernel";
    }
    /** Default constructor */
    NEL2NormalizeLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEL2NormalizeLayerKernel(const NEL2NormalizeLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEL2NormalizeLayerKernel &operator=(const NEL2NormalizeLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEL2NormalizeLayerKernel(NEL2NormalizeLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEL2NormalizeLayerKernel &operator=(NEL2NormalizeLayerKernel &&) = default;
    /** Default destructor */
    ~NEL2NormalizeLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data types supported: F16/F32.
     * @param[in]  sum     Sum values tensor. Data types supported: same as @p input.
     *                     Sum will have the same number of dimensions as input.
     * @param[out] output  Destination tensor. Data types and data layouts supported: same as @p input.
     *                     Output will have the same number of dimensions as input.
     * @param[in]  axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in]  epsilon Lower bound value for the normalization.
     */
    void configure(const ITensor *input, const ITensor *sum, ITensor *output, int axis, float epsilon);

    /** Static function to check if given info will lead to a valid configuration of @ref NEL2NormalizeLayerKernel.
     *
     * @param[in] input   Source tensor info. Data types supported: F16/F32.
     * @param[in] sum     Sum values tensor info. Data types supported: same as @p input.
     *                    Sum will have the same number of dimensions as input.
     * @param[in] output  Destination tensor info. Data types and data layouts supported: same as @p input.
     *                    Output will have the same number of dimensions as input.
     * @param[in] axis    Axis along which to reduce. Negative values wrap around. Maximum supported actual reduction axis : 2
     * @param[in] epsilon Lower bound value for the normalization.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    const ITensor *_sum;
    ITensor       *_output;
    unsigned int   _actual_axis;
    float          _epsilon;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEL2NORMALIZELAYERKERNEL_H */
