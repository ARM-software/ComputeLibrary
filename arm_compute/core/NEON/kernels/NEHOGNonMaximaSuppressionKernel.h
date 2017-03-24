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
#ifndef __ARM_COMPUTE_NEHOGNONMAXIMASUPPRESSIONKERNEL_H__
#define __ARM_COMPUTE_NEHOGNONMAXIMASUPPRESSIONKERNEL_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/IHOG.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** NEON kernel to perform in-place computation of euclidean distance based non-maxima suppression for HOG
 *
 * @note This kernel is meant to be used alongside HOG and performs a non-maxima suppression on a
 *       HOG detection window.
 */
class NEHOGNonMaximaSuppressionKernel : public INEKernel
{
public:
    /** Default constructor */
    NEHOGNonMaximaSuppressionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGNonMaximaSuppressionKernel(const NEHOGNonMaximaSuppressionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGNonMaximaSuppressionKernel &operator=(const NEHOGNonMaximaSuppressionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEHOGNonMaximaSuppressionKernel(NEHOGNonMaximaSuppressionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEHOGNonMaximaSuppressionKernel &operator=(NEHOGNonMaximaSuppressionKernel &&) = default;
    /** Initialise the kernel's input, output and the euclidean minimum distance
     *
     * @param[in, out] input_output Input/Output array of @ref DetectionWindow
     * @param[in]      min_distance Radial Euclidean distance for non-maxima suppression
     */
    void configure(IDetectionWindowArray *input_output, float min_distance);

    // Inherited methods overridden:
    void run(const Window &window) override;
    bool is_parallelisable() const override;

private:
    IDetectionWindowArray *_input_output;
    float                  _min_distance;
};
}

#endif /* __ARM_COMPUTE_NEHOGNONMAXIMASUPPRESSIONKERNEL_H__ */
