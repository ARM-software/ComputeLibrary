/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGAUSSIANPYRAMIDKERNEL_H
#define ARM_COMPUTE_NEGAUSSIANPYRAMIDKERNEL_H

#include "src/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform a GaussianPyramid (horizontal pass) */
class NEGaussianPyramidHorKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGaussianPyramidHorKernel";
    }
    /** Default constructor */
    NEGaussianPyramidHorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidHorKernel(NEGaussianPyramidHorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidHorKernel &operator=(NEGaussianPyramidHorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidHorKernel(NEGaussianPyramidHorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidHorKernel &operator=(NEGaussianPyramidHorKernel &&) = default;
    /** Default destructor */
    ~NEGaussianPyramidHorKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input  Source tensor. Data type supported: U8.
     * @param[out] output Destination tensor. Output should have half the input width. Data type supported: S16.
     */
    void configure(const ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    int _l2_load_offset;
};

/** Neon kernel to perform a GaussianPyramid (vertical pass) */
class NEGaussianPyramidVertKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGaussianPyramidVertKernel";
    }
    /** Default constructor */
    NEGaussianPyramidVertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidVertKernel(NEGaussianPyramidVertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussianPyramidVertKernel &operator=(NEGaussianPyramidVertKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidVertKernel(NEGaussianPyramidVertKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussianPyramidVertKernel &operator=(NEGaussianPyramidVertKernel &&) = default;
    /** Default destructor */
    ~NEGaussianPyramidVertKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input  Source tensor. Data type supported: S16.
     * @param[out] output Destination tensor. Output should have half the input height. Data type supported: U8.
     */
    void configure(const ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    int _t2_load_offset;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGAUSSIANPYRAMIDKERNEL_H */
