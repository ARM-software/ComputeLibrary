/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_NORMALIZATION_TYPE_DATASET_H__
#define __ARM_COMPUTE_TEST_NORMALIZATION_TYPE_DATASET_H__

#include "arm_compute/core/Types.h"

#ifdef BOOST
#include "boost_wrapper.h"
#endif

namespace arm_compute
{
namespace test
{
/** Data set containing all possible normalization types.
 *
 * Can be used as input for Boost data test cases to automatically run a test
 * case on all normalization types.
 */
class NormalizationTypes
{
public:
    /** Type of the samples in the data set. */
    using sample = NormType;

    /** Dimensionality of the data set. */
    enum
    {
        arity = 1
    };

    /** Number of samples in the data set. */
#ifdef BOOST
    boost::unit_test::data::size_t size() const
#else
    unsigned int size() const
#endif
    {
        return _types.size();
    }

    /** Type of the iterator used to step through all samples in the data set.
     * Needs to support operator*() and operator++() which a pointer does.
     */
    using iterator = const NormType *;

    /** Iterator to the first sample in the data set. */
    iterator begin() const
    {
        return _types.data();
    }

private:
    std::array<NormType, 3> _types{ { NormType::IN_MAP_1D, NormType::IN_MAP_2D, NormType::CROSS_MAP } };
};
} // namespace test
} // namespace arm_compute
#endif
