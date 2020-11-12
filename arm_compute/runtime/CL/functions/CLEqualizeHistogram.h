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
#ifndef ARM_COMPUTE_CLEQUALIZEHISTOGRAM_H
#define ARM_COMPUTE_CLEQUALIZEHISTOGRAM_H

#include "arm_compute/runtime/CL/CLDistribution1D.h"
#include "arm_compute/runtime/CL/CLLut.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class CLCompileContext;
class CLHistogramKernel;
class CLHistogramBorderKernel;
class CLTableLookupKernel;
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to execute histogram equalization. This function calls the following CL kernels:
 *
 * -# @ref CLHistogramKernel
 * -# @ref CLTableLookupKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class CLEqualizeHistogram : public IFunction
{
public:
    /** Default Constructor. */
    CLEqualizeHistogram();
    /** Prevent instances of this class from being copied */
    CLEqualizeHistogram(const CLEqualizeHistogram &) = delete;
    /** Prevent instances of this class from being copied */
    CLEqualizeHistogram &operator=(const CLEqualizeHistogram &) = delete;
    /** Default destructor */
    ~CLEqualizeHistogram();
    /** Initialise the kernel's inputs.
     *
     * @param[in]  input  Input image. Data types supported: U8.
     * @param[out] output Output of same data type with equalized brightness and contrast.
     */
    void configure(const ICLImage *input, ICLImage *output);
    /** Initialise the kernel's inputs.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input image. Data types supported: U8.
     * @param[out] output          Output of same data type with equalized brightness and contrast.
     */
    void configure(const CLCompileContext &compile_context, const ICLImage *input, ICLImage *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<CLHistogramKernel>       _histogram_kernel;        /**< Kernel that calculates the histogram of input. */
    std::unique_ptr<CLHistogramBorderKernel> _border_histogram_kernel; /**< Kernel that calculates the histogram on the borders. */
    std::unique_ptr<CLTableLookupKernel>     _map_histogram_kernel;    /**< Kernel that maps the input to output using the lut. */
    CLDistribution1D                         _hist;                    /**< Distribution that holds the histogram of the input image. */
    CLDistribution1D                         _cum_dist;                /**< Distribution that holds the cummulative distribution of the input histogram. */
    CLLut                                    _cd_lut;                  /**< Holds the equalization lookuptable. */
    static const uint32_t                    max_range = 256;          /**< Histogram range of the internal histograms. */
    static const uint32_t                    nr_bins   = 256;          /**< Histogram bins of the internal histograms. */
};
}
#endif /*ARM_COMPUTE_CLEQUALIZEHISTOGRAM_H */
