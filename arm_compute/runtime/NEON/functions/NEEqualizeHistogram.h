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
#ifndef ARM_COMPUTE_NEEQUALIZEHISTOGRAM_H
#define ARM_COMPUTE_NEEQUALIZEHISTOGRAM_H

#include "arm_compute/runtime/Distribution1D.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/Lut.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
class NEHistogramKernel;
class NECumulativeDistributionKernel;
class NETableLookupKernel;
using IImage = ITensor;

/** Basic function to execute histogram equalization. This function calls the following Neon kernels:
 *
 * -# @ref NEHistogramKernel
 * -# @ref NECumulativeDistributionKernel
 * -# @ref NETableLookupKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEEqualizeHistogram : public IFunction
{
public:
    /** Default Constructor. */
    NEEqualizeHistogram();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEqualizeHistogram(const NEEqualizeHistogram &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEEqualizeHistogram &operator=(const NEEqualizeHistogram &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEEqualizeHistogram(NEEqualizeHistogram &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEEqualizeHistogram &operator=(NEEqualizeHistogram &&) = delete;
    /** Default destructor */
    ~NEEqualizeHistogram();
    /** Initialise the kernel's inputs.
     *
     * @note Currently the width of the input image must be a multiple of 16.
     *
     * @param[in]  input  Input image. Data type supported: U8.
     * @param[out] output Output image. Data type supported: same as @p input
     */
    void configure(const IImage *input, IImage *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEHistogramKernel>              _histogram_kernel;        /**< Kernel that calculates the histogram of input. */
    std::unique_ptr<NECumulativeDistributionKernel> _cd_histogram_kernel;     /**< Kernel that calculates the cumulative distribution
                                                                  and creates the relevant LookupTable. */
    std::unique_ptr<NETableLookupKernel>            _map_histogram_kernel;    /**< Kernel that maps the input to output using the lut. */
    Distribution1D                                  _hist;                    /**< Distribution that holds the histogram of the input image. */
    Distribution1D                                  _cum_dist;                /**< Distribution that holds the cummulative distribution of the input histogram. */
    Lut                                             _cd_lut;                  /**< Holds the equalization lookuptable. */
    static constexpr uint32_t                       nr_bins{ 256 };           /**< Histogram bins of the internal histograms. */
    static constexpr uint32_t                       max_range{ nr_bins - 1 }; /**< Histogram range of the internal histograms. */
};
}
#endif /*ARM_COMPUTE_NEEQUALIZEHISTOGRAM_H */
