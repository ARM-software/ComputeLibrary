/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEHISTOGRAM_H__
#define __ARM_COMPUTE_NEHISTOGRAM_H__

#include "arm_compute/core/NEON/kernels/NEHistogramKernel.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace arm_compute
{
class IDistribution1D;

/** Basic function to run @ref NEHistogramKernel. */
class NEHistogram : public IFunction
{
public:
    /** Default Constructor. */
    NEHistogram();
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input  Input image. Data type supported: U8.
     * @param[out] output Output distribution.
     */
    void configure(const IImage *input, IDistribution1D *output);

    // Inherited methods overridden:
    void run() override;

private:
    NEHistogramKernel     _histogram_kernel;
    std::vector<uint32_t> _local_hist;
    std::vector<uint32_t> _window_lut;
    size_t                _local_hist_size;
    /** 256 possible pixel values as we handle only U8 images */
    static constexpr unsigned int window_lut_default_size = 256;
};
}
#endif /*__ARM_COMPUTE_NEHISTOGRAM_H__ */
