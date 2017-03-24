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
#ifndef __ARM_COMPUTE_IDISTRIBUTION1D_H__
#define __ARM_COMPUTE_IDISTRIBUTION1D_H__

#include "arm_compute/core/IDistribution.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
/** 1D Distribution interface */
class IDistribution1D : public IDistribution
{
public:
    /** Constructor: Creates a 1D Distribution of a consecutive interval [offset, offset + range - 1]
     *               defined by a start offset and valid range, divided equally into num_bins parts.
     *
     * @param[in] num_bins The number of bins the distribution is divided in.
     * @param[in] offset   The start of the values to use.
     * @param[in] range    The total number of the consecutive values of the distribution interval.
     */
    IDistribution1D(size_t num_bins, int32_t offset, uint32_t range);
    /** Returns the number of bins that the distribution has.
     *
     * @return Number of bins of the distribution.
     */
    size_t num_bins() const;
    /** Returns the offset of the distribution.
     *
     * @return Offset of the distribution.
     */
    int32_t offset() const;
    /** Returns the range of the distribution.
     *
     * @return Range of the distribution.
     */
    uint32_t range() const;
    /** Returns the window of the distribution, which is the range divided by the number of bins.
     *
     * @note If range is not divided by the number of bins then it is invalid.
     *
     * @return Window of the distribution.
     */
    uint32_t window() const;
    /** Sets the range of the distribution.
     *
     * @param[in] range New range of the distribution to be set.
     */
    void set_range(uint32_t range);

    // Inherited methods overridden:
    size_t size() const override;
    size_t dimensions() const override;

private:
    size_t   _num_bins; /**< Number of bins. */
    int32_t  _offset;   /**< Offset, which indicate the start of the usable values. */
    uint32_t _range;    /**< The total number of consecutive values of the distribution interval */
};
}
#endif /* __ARM_COMPUTE_IDISTRIBUTION1D_H__ */
