/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NECHANNELCOMBINE_H__
#define __ARM_COMPUTE_NECHANNELCOMBINE_H__

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class IMultiImage;
class ITensor;
using IImage = ITensor;

/**Basic function to run @ref NEChannelCombineKernel to perform channel combination. */
class NEChannelCombine : public INESimpleFunctionNoBorder
{
public:
    /** Initialize function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Data type supported: U8
     * @param[in]  plane1 The 2D plane that forms channel 1. Data type supported: U8
     * @param[in]  plane2 The 2D plane that forms channel 2. Data type supported: U8
     * @param[in]  plane3 The 2D plane that forms channel 3. Data type supported: U8
     * @param[out] output The single planar output tensor. Formats supported: RGB888/RGBA8888/UYVY422/YUYV422
     */
    void configure(const ITensor *plane0, const ITensor *plane1, const ITensor *plane2, const ITensor *plane3, ITensor *output);
    /** Initialize function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Data type supported: U8
     * @param[in]  plane1 The 2D plane that forms channel 1. Data type supported: U8
     * @param[in]  plane2 The 2D plane that forms channel 2. Data type supported: U8
     * @param[out] output The multi planar output image. Formats supported: NV12/NV21/IYUV/YUV444
     */
    void configure(const IImage *plane0, const IImage *plane1, const IImage *plane2, IMultiImage *output);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECHANNELCOMBINE_H__*/
