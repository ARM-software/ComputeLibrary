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
#ifndef ARM_COMPUTE_NEGAUSSIAN5x5KERNEL_H
#define ARM_COMPUTE_NEGAUSSIAN5x5KERNEL_H

#include "src/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform a Gaussian 5x5 filter (horizontal pass) */
class NEGaussian5x5HorKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGaussian5x5HorKernel";
    }
    /** Default constructor */
    NEGaussian5x5HorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5HorKernel(NEGaussian5x5HorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5HorKernel &operator=(NEGaussian5x5HorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussian5x5HorKernel(NEGaussian5x5HorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussian5x5HorKernel &operator=(NEGaussian5x5HorKernel &&) = default;
    /** Default destructor */
    ~NEGaussian5x5HorKernel() = default;

    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: U8.
     * @param[out] output           Destination tensor. Data type supported: S16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    BorderSize _border_size;
};

/** Neon kernel to perform a Gaussian 5x5 filter (vertical pass) */
class NEGaussian5x5VertKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGaussian5x5VertKernel";
    }
    /** Default constructor */
    NEGaussian5x5VertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5VertKernel(NEGaussian5x5VertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGaussian5x5VertKernel &operator=(NEGaussian5x5VertKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGaussian5x5VertKernel(NEGaussian5x5VertKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGaussian5x5VertKernel &operator=(NEGaussian5x5VertKernel &&) = default;
    /** Default destructor */
    ~NEGaussian5x5VertKernel() = default;
    /** Initialise the kernel's source, destination and border mode.
     *
     * @param[in]  input            Source tensor. Data type supported: S16.
     * @param[out] output           Destination tensor, Data type supported: U8.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ITensor *input, ITensor *output, bool border_undefined);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGAUSSIAN5x5KERNEL_H */
