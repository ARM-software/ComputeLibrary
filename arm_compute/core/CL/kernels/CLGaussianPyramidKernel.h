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
#ifndef __ARM_COMPUTE_CLGAUSSIANPYRAMIDKERNEL_H__
#define __ARM_COMPUTE_CLGAUSSIANPYRAMIDKERNEL_H__

#include "arm_compute/core/CL/ICLSimpleKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform a Gaussian filter and half scaling across width (horizontal pass) */
class CLGaussianPyramidHorKernel : public ICLSimpleKernel
{
public:
    /** Default constructor */
    CLGaussianPyramidHorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramidHorKernel(const CLGaussianPyramidHorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramidHorKernel &operator=(const CLGaussianPyramidHorKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGaussianPyramidHorKernel(CLGaussianPyramidHorKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGaussianPyramidHorKernel &operator=(CLGaussianPyramidHorKernel &&) = default;
    /** Default destructor */
    ~CLGaussianPyramidHorKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output           Destination tensor. Output should have half the input width. Data types supported: U16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    BorderSize _border_size;
    int        _l2_load_offset;
};

/** OpenCL kernel to perform a Gaussian filter and half scaling across height (vertical pass) */
class CLGaussianPyramidVertKernel : public ICLSimpleKernel
{
public:
    /** Default constructor */
    CLGaussianPyramidVertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramidVertKernel(const CLGaussianPyramidVertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGaussianPyramidVertKernel &operator=(const CLGaussianPyramidVertKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGaussianPyramidVertKernel(CLGaussianPyramidVertKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGaussianPyramidVertKernel &operator=(CLGaussianPyramidVertKernel &&) = default;
    /** Default destructor */
    ~CLGaussianPyramidVertKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input            Source tensor. Data types supported: U16.
     * @param[out] output           Destination tensor. Output should have half the input height. Data types supported: U8.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    int _t2_load_offset;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGAUSSIANPYRAMIDKERNEL_H__ */
