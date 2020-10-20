/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEHISTOGRAM_H
#define ARM_COMPUTE_NEHISTOGRAM_H

#include "arm_compute/runtime/IFunction.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class ITensor;
class IDistribution1D;
class NEHistogramKernel;
using IImage = ITensor;

/** Basic function to run @ref NEHistogramKernel. */
class NEHistogram : public IFunction
{
public:
    /** Default Constructor. */
    NEHistogram();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHistogram(const NEHistogram &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHistogram &operator=(const NEHistogram &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHistogram(NEHistogram &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHistogram &operator=(NEHistogram &&) = delete;
    /** Default destructor */
    ~NEHistogram();
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input  Input image. Data type supported: U8.
     * @param[out] output Output distribution.
     */
    void configure(const IImage *input, IDistribution1D *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEHistogramKernel> _histogram_kernel;
    std::vector<uint32_t>              _local_hist;
    std::vector<uint32_t>              _window_lut;
    size_t                             _local_hist_size;
    /** 256 possible pixel values as we handle only U8 images */
    static constexpr unsigned int window_lut_default_size = 256;
};
}
#endif /*ARM_COMPUTE_NEHISTOGRAM_H */
