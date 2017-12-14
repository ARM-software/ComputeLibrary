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
#ifndef __ARM_COMPUTE_CLL2NORMALIZEKERNEL_H__
#define __ARM_COMPUTE_CLL2NORMALIZEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the reduction operation kernel */
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
     * @param[in]  input   Source tensor. Data types supported: QS8, QS16, F32.
     * @param[in]  sum     Sum values tensor. Data types supported: same as @p input.
     * @param[out] output  Destination tensor. Data types supported: Same as @p input.
     * @param[in]  axis    Axis along which to reduce. Supported reduction axis : 0
     * @param[in]  epsilon Lower bound value for the normalization.
     */
    void configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, unsigned int axis, float epsilon);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_sum;
    ICLTensor       *_output;
    unsigned int     _axis;
    float            _epsilon;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLL2NORMALIZEKERNEL_H__ */
