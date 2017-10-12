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
#ifndef ARM_COMPUTE_TEST_DATASET_CARTESIAN_PRODUCT
#define ARM_COMPUTE_TEST_DATASET_CARTESIAN_PRODUCT

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
/** Implementation of a dataset representing all combinations of values of the input datasets.
 *
 * For example, for the inputs {1, 2} and {3, 4} this dataset virtually
 * represents the values {(1, 3), (1, 4), (2, 3), (2, 4)}.
 */
template <typename T, typename U>
class CartesianProductDataset : public Dataset
{
private:
    using T_noref    = typename std::remove_reference<T>::type;
    using U_noref    = typename std::remove_reference<U>::type;
    using iter1_type = typename T_noref::iterator;
    using iter2_type = typename U_noref::iterator;

public:
    /** Construct dataset from the given datasets.
     *
     * @param[in] dataset1 First dataset.
     * @param[in] dataset2 Second dataset.
     */
    CartesianProductDataset(T &&dataset1, U &&dataset2)
        : _dataset1{ std::forward<T>(dataset1) },
          _dataset2{ std::forward<U>(dataset2) }
    {
    }

    CartesianProductDataset(CartesianProductDataset &&) = default;

    /** Type of the dataset. */
    using type = decltype(std::tuple_cat(*std::declval<iter1_type>(), *std::declval<iter2_type>()));

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(const T_noref *dataset1, const U_noref *dataset2)
            : _iter1{ dataset1->begin() },
              _dataset2{ dataset2 },
              _iter2{ dataset2->begin() }
        {
        }

        iterator(const iterator &) = default;
        iterator &operator=(const iterator &) = default;
        iterator(iterator &&)                 = default;
        iterator &operator=(iterator &&) = default;

        ~iterator() = default;

        std::string description() const
        {
            return _iter1.description() + ":" + _iter2.description();
        }

        CartesianProductDataset::type operator*() const
        {
            return std::tuple_cat(*_iter1, *_iter2);
        }

        iterator &operator++()
        {
            ++_second_pos;

            if(_second_pos < _dataset2->size())
            {
                ++_iter2;
            }
            else
            {
                _second_pos = 0;
                _iter2      = _dataset2->begin();

                ++_iter1;
            }

            return *this;
        }

    private:
        iter1_type     _iter1;
        const U_noref *_dataset2;
        iter2_type     _iter2;
        int            _first_pos{ 0 };
        int            _second_pos{ 0 };
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(&_dataset1, &_dataset2);
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return _dataset1.size() * _dataset2.size();
    }

private:
    T _dataset1;
    U _dataset2;
};

/** Helper function to create a @ref CartesianProductDataset.
 *
 * @param[in] dataset1 First dataset.
 * @param[in] dataset2 Second dataset.
 *
 * @return A grid dataset.
 */
template <typename T, typename U>
CartesianProductDataset<T, U> combine(T &&dataset1, U &&dataset2)
{
    return CartesianProductDataset<T, U>(std::forward<T>(dataset1), std::forward<U>(dataset2));
}

template <typename T, typename U>
CartesianProductDataset<T, U>
operator*(T &&dataset1, U &&dataset2)
{
    return CartesianProductDataset<T, U>(std::forward<T>(dataset1), std::forward<U>(dataset2));
}

} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_CARTESIAN_PRODUCT */
