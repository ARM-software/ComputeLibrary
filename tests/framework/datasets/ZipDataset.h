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
#ifndef ARM_COMPUTE_TEST_DATASET_ZIP
#define ARM_COMPUTE_TEST_DATASET_ZIP

#include "Dataset.h"

#include <string>
#include <tuple>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace dataset
{
/** Implementation of a dataset representing pairs of values of the input datasets.
 *
 * For example, for the inputs {1, 2} and {3, 4} this dataset virtually
 * represents the values {(1, 3), (1, 4)}.
 */
template <typename T, typename U>
class ZipDataset : public Dataset
{
private:
    using iter1_type = typename T::iterator;
    using iter2_type = typename U::iterator;

public:
    /** Construct dataset from the given datasets.
     *
     * @param[in] dataset1 First dataset.
     * @param[in] dataset2 Second dataset.
     */
    ZipDataset(T &&dataset1, U &&dataset2)
        : _dataset1{ std::forward<T>(dataset1) },
          _dataset2{ std::forward<U>(dataset2) }
    {
    }

    ZipDataset(ZipDataset &&) = default;

    /** Type of the dataset. */
    using type = decltype(std::tuple_cat(*std::declval<iter1_type>(), *std::declval<iter2_type>()));

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(iter1_type iter1, iter2_type iter2)
            : _iter1{ std::move(iter1) }, _iter2{ std::move(iter2) }
        {
        }

        std::string description() const
        {
            return _iter1.description() + ":" + _iter2.description();
        }

        ZipDataset::type operator*() const
        {
            return std::tuple_cat(*_iter1, *_iter2);
        }

        iterator &operator++()
        {
            ++_iter1;
            ++_iter2;
            return *this;
        }

    private:
        iter1_type _iter1;
        iter2_type _iter2;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(_dataset1.begin(), _dataset2.begin());
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return std::min(_dataset1.size(), _dataset2.size());
    }

private:
    T _dataset1;
    U _dataset2;
};

/** Helper function to create a @ref ZipDataset.
 *
 * @param[in] dataset1 First dataset.
 * @param[in] dataset2 Second dataset.
 *
 * @return A zip dataset.
 */
template <typename T, typename U>
ZipDataset<T, U> zip(T &&dataset1, U &&dataset2)
{
    return ZipDataset<T, U>(std::forward<T>(dataset1), std::forward<U>(dataset2));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_ZIP */
