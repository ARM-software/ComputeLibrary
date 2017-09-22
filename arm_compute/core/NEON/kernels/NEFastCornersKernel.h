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
#ifndef __ARM_COMPUTE_NEFASTCORNERSKERNEL_H__
#define __ARM_COMPUTE_NEFASTCORNERSKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** NEON kernel to perform fast corners */
class NEFastCornersKernel : public INEKernel
{
public:
    /** Constructor */
    NEFastCornersKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFastCornersKernel(const NEFastCornersKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFastCornersKernel &operator=(const NEFastCornersKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFastCornersKernel(NEFastCornersKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFastCornersKernel &operator=(NEFastCornersKernel &&) = default;
    /** Initialise the kernel.
     *
     * @param[in]  input               Source image. Data type supported: U8.
     * @param[out] output              Output image. Data type supported: U8.
     * @param[in]  threshold           Threshold on difference between intensity of the central pixel and pixels on Bresenham's circle of radius 3.
     * @param[in]  non_max_suppression True if non-maxima suppresion is applied, false otherwise.
     * @param[in]  border_undefined    True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const IImage *input, IImage *output, uint8_t threshold, bool non_max_suppression, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    const IImage *_input;               /**< source image */
    IImage       *_output;              /**< inermediate results */
    uint8_t       _threshold;           /**< threshold on difference between intensity */
    bool          _non_max_suppression; /** true if non-maxima suppression is applied in the next stage */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEFASTCORNERSKERNEL_H__ */
