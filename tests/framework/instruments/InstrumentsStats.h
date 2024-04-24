/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_INSTRUMENTSMAP
#define ARM_COMPUTE_TEST_INSTRUMENTSMAP

#include "Measurement.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Generate common statistics for a set of measurements
     */
class InstrumentsStats
{
public:
    /** Compute statistics for the passed set of measurements
     *
     * @param[in] measurements The measurements to process
     */
    InstrumentsStats(const std::vector<Measurement> &measurements);
    /** The measurement with the minimum value
             */
    const Measurement &min() const
    {
        return *_min;
    }
    /** The measurement with the maximum value
             */
    const Measurement &max() const
    {
        return *_max;
    }
    /** The median measurement
             */
    const Measurement &median() const
    {
        return *_median;
    }
    /** The average of all the measurements
             */
    const Measurement::Value &mean() const
    {
        return _mean;
    }
    /** The relative standard deviation of the measurements
             */
    double relative_standard_deviation() const
    {
        return _stddev;
    }

private:
    const Measurement *_min;
    const Measurement *_max;
    const Measurement *_median;
    Measurement::Value _mean;
    double             _stddev;
};

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTRUMENTSMAP */
