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
#ifndef __ARM_COMPUTE_CLCOLORCONVERTKERNEL_H__
#define __ARM_COMPUTE_CLCOLORCONVERTKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLMultiImage;
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for the color convert kernel.
 *
 */
class CLColorConvertKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLColorConvertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLColorConvertKernel(const CLColorConvertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLColorConvertKernel &operator=(const CLColorConvertKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLColorConvertKernel(CLColorConvertKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLColorConvertKernel &operator=(CLColorConvertKernel &&) = default;
    /** Default destructor. */
    ~CLColorConvertKernel() = default;

    /** Set the input and output of the kernel
     *
     * @param[in]  input  Source tensor
     * @param[out] output Destination tensor
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Set the input and output of the kernel
     *
     * @param[in]  input  multi-planar source image
     * @param[out] output single-planar destination image
     */
    void configure(const ICLMultiImage *input, ICLImage *output);
    /** Set the input and output of the kernel
     *
     * @param[in]  input  single-planar source image
     * @param[out] output multi-planar destination image
     */
    void configure(const ICLImage *input, ICLMultiImage *output);
    /** Set the input and output of the kernel
     *
     * @param[in]  input  multi-planar source image
     * @param[out] output multi-planar destination image
     */
    void configure(const ICLMultiImage *input, ICLMultiImage *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor     *_input;        /*pointer to single planar tensor input */
    ICLTensor           *_output;       /*pointer to single planar tensor output */
    const ICLMultiImage *_multi_input;  /*pointer to multi-planar input */
    ICLMultiImage       *_multi_output; /*pointer to multi-planar output */
};
}

#endif /* __ARM_COMPUTE_CLCOLORCONVERTKERNEL_H__ */
