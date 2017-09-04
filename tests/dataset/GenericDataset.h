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
#ifndef __ARM_COMPUTE_TEST_DATASET_GENERIC_DATASET_H__
#define __ARM_COMPUTE_TEST_DATASET_GENERIC_DATASET_H__

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif

namespace arm_compute
{
namespace test
{
/** Abstract data set containing multiple objects T.
 *
 * Can be used as input for Boost data test cases to automatically run a test
 * case on different configurations.
 */
template <class T, unsigned int Size>
class GenericDataset
{
public:
    /** Type of the samples in the data set. */
    using sample = T;

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
        return _data.size();
    }

    /** Type of the iterator used to step through all samples in the data set.
     * Needs to support operator*() and operator++() which a pointer does.
     */
    using iterator = const T *;

    /** Iterator to the first sample in the data set. */
    iterator begin() const
    {
        return _data.data();
    }

protected:
    /** Protected constructor to make the class abstract. */
    template <typename... Ts>
    GenericDataset(Ts... objs)
        : _data{ { objs... } }
    {
    }

    /** Protected destructor to prevent deletion of derived class through a
     * pointer to the base class.
     */
    ~GenericDataset() = default;

private:
    std::array<T, Size> _data;
};
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_DATASET_GENERIC_DATASET_H__
