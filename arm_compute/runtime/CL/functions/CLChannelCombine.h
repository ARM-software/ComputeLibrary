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
#ifndef __ARM_COMPUTE_CLCHANNELCOMBINE_H__
#define __ARM_COMPUTE_CLCHANNELCOMBINE_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLMultiImage;
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to run @ref CLChannelCombineKernel to perform channel combination. */
class CLChannelCombine : public ICLSimpleFunction
{
public:
    /** Initialize function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Must be of U8 format.
     * @param[in]  plane1 The 2D plane that forms channel 1. Must be of U8 format.
     * @param[in]  plane2 The 2D plane that forms channel 2. Must be of U8 format.
     * @param[in]  plane3 The 2D plane that forms channel 3. Must be of U8 format.
     * @param[out] output The single planar output tensor.
     */
    void configure(const ICLTensor *plane0, const ICLTensor *plane1, const ICLTensor *plane2, const ICLTensor *plane3, ICLTensor *output);
    /** Initialize function's inputs and outputs.
     *
     * @param[in]  plane0 The 2D plane that forms channel 0. Must be of U8 format.
     * @param[in]  plane1 The 2D plane that forms channel 1. Must be of U8 format.
     * @param[in]  plane2 The 2D plane that forms channel 2. Must be of U8 format.
     * @param[out] output The multi planar output image.
     */
    void configure(const ICLImage *plane0, const ICLImage *plane1, const ICLImage *plane2, ICLMultiImage *output);
};
}
#endif /*__ARM_COMPUTE_CLCHANNELCOMBINE_H__*/
