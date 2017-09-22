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
#ifndef __ARM_COMPUTE_NEFILLARRAYKERNEL_H__
#define __ARM_COMPUTE_NEFILLARRAYKERNEL_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** This kernel adds all texels greater than or equal to the threshold value to the keypoint array. */
class NEFillArrayKernel : public INEKernel
{
public:
    /** Default contructor */
    NEFillArrayKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillArrayKernel(const NEFillArrayKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFillArrayKernel &operator=(const NEFillArrayKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEFillArrayKernel(NEFillArrayKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEFillArrayKernel &operator=(NEFillArrayKernel &&) = default;
    /** Default detructor */
    ~NEFillArrayKernel() = default;

    /** Initialise the kernel.
     *
     * @param[in]  input     Source image. Data type supported: U8.
     * @param[in]  threshold Texels greater than the threshold will be added to the array.
     * @param[out] output    Arrays of keypoints to store the results.
     */
    void configure(const IImage *input, uint8_t threshold, IKeyPointArray *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    const IImage   *_input;
    IKeyPointArray *_output;
    uint8_t         _threshold;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEFILLARRAYKERNEL_H__*/
