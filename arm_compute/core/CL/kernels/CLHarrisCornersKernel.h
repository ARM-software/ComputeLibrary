/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLHARRISCORNERSKERNEL_H__
#define __ARM_COMPUTE_CLHARRISCORNERSKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for the harris score kernel.
 *
 * @note The implementation supports 3, 5, and 7 for the block_size.
 */
class CLHarrisScoreKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLHarrisScoreKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHarrisScoreKernel(const CLHarrisScoreKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHarrisScoreKernel &operator=(const CLHarrisScoreKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHarrisScoreKernel(CLHarrisScoreKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHarrisScoreKernel &operator=(CLHarrisScoreKernel &&) = default;
    /** Default destructor */
    ~CLHarrisScoreKernel() = default;

    /** Setup the kernel parameters
     *
     * @param[in]  input1           Source image (gradient X). Data types supported S16, S32. (Must be the same as input2)
     * @param[in]  input2           Source image (gradient Y). Data types supported S16, S32. (Must be the same as input1)
     * @param[out] output           Destination image (harris score). Data types supported F32
     * @param[in]  block_size       The block window size used to compute the Harris Corner score.  Supports: 3, 5 and 7
     * @param[in]  norm_factor      Normalization factor to use accordingly with the gradient size (Must be different from 0)
     * @param[in]  strength_thresh  Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
     * @param[in]  sensitivity      Sensitivity threshold k from the Harris-Stephens equation.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLImage *input1, const ICLImage *input2, ICLImage *output,
                   int32_t block_size, float norm_factor, float strength_thresh, float sensitivity,
                   bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

protected:
    const ICLImage *_input1;          /**< Source image - Gx component */
    const ICLImage *_input2;          /**< Source image - Gy component */
    ICLImage       *_output;          /**< Source image - Harris score */
    float           _sensitivity;     /**< Sensitivity value */
    float           _strength_thresh; /**< Threshold value */
    float           _norm_factor;     /**< Normalization factor */
    BorderSize      _border_size;     /**< Border size */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLHARRISCORNERSKERNEL_H__ */
