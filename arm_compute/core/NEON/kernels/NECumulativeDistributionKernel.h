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
#ifndef __ARM_COMPUTE_NECUMULATIVEDISTRIBUTIONKERNEL_H__
#define __ARM_COMPUTE_NECUMULATIVEDISTRIBUTIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include <cstdint>

namespace arm_compute
{
class IDistribution1D;
class ILut;
class ITensor;
using IImage = ITensor;

/** Interface for the cumulative distribution (cummulative summmation) calculation kernel.
 *
 * This kernel calculates the cumulative sum of a given distribution (meaning that each output element
 * is the sum of all its previous elements including itself) and creates a lookup table with the normalized
 * pixel intensities which is used for improve the constrast of the image.
 */
class NECumulativeDistributionKernel : public INEKernel
{
public:
    /** Default constructor */
    NECumulativeDistributionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECumulativeDistributionKernel(const NECumulativeDistributionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECumulativeDistributionKernel &operator=(const NECumulativeDistributionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NECumulativeDistributionKernel(NECumulativeDistributionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NECumulativeDistributionKernel &operator=(NECumulativeDistributionKernel &&) = default;
    /** Set the input and output distribution.
     *
     * @param[in]  input          Input image. Data type supported: U8
     * @param[in]  distribution   Unnormalized 256-bin distribution of the input image.
     * @param[out] cumulative_sum Cummulative distribution (Summed histogram). Should be same size as @p distribution.
     * @param[out] output         Equalization lookup table. Should consist of 256 entries of U8 elements.
     */
    void configure(const IImage *input, const IDistribution1D *distribution, IDistribution1D *cumulative_sum, ILut *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    const IImage          *_input;          /**< Input image. */
    const IDistribution1D *_distribution;   /**< Input histogram of the input image. */
    IDistribution1D       *_cumulative_sum; /**< The cummulative distribution. */
    ILut                  *_output;         /**< Output with the equalization lookup table. */
private:
    static const uint32_t _histogram_size = 256; /**< Default histogram size of 256. */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECUMULATIVEDISTRIBUTIONKERNEL_H__ */
