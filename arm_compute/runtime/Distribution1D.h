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
#ifndef __ARM_COMPUTE_DISTRIBUTION1D_H__
#define __ARM_COMPUTE_DISTRIBUTION1D_H__

#include "arm_compute/core/IDistribution1D.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
/** Basic implementation of the 1D distribution interface */
class Distribution1D : public IDistribution1D
{
public:
    /** Constructor: Creates a 1D Distribution of a consecutive interval [offset, offset + range - 1]
     *               defined by a start offset and valid range, divided equally into num_bins parts.
     *
     * @param[in] num_bins The number of bins the distribution is divided in.
     * @param[in] offset   The start of the values to use.
     * @param[in] range    The total number of the consecutive values of the distribution interval.
     */
    Distribution1D(size_t num_bins, int32_t offset, uint32_t range);

    // Inherited methods overridden:
    uint32_t *buffer() const override;

private:
    mutable std::vector<uint32_t> _data; /**< The distribution data. */
};
}
#endif /* __ARM_COMPUTE_DISTRIBUTION1D_H__ */
