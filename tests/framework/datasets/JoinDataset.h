/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DATASET_JOIN
#define ARM_COMPUTE_TEST_DATASET_JOIN

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
/** Implementation of a dataset representing the concatenation of the input datasets.
 *
 * For example, for the inputs {1, 2} and {3, 4} this dataset virtually
 * represents the values {1, 2, 3, 4}.
 */
template <typename T, typename U>
class JoinDataset : public Dataset
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
    JoinDataset(T &&dataset1, U &&dataset2)
        : _dataset1{ std::forward<T>(dataset1) },
          _dataset2{ std::forward<U>(dataset2) }
    {
    }

    JoinDataset(JoinDataset &&) = default;

    /** Type of the dataset. */
    using type = typename T_noref::type;

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(const T_noref *dataset1, const U_noref *dataset2)
            : _iter1{ dataset1->begin() }, _iter2{ dataset2->begin() }, _first_size{ dataset1->size() }
        {
        }

        std::string description() const
        {
            return _first_size > 0 ? _iter1.description() : _iter2.description();
        }

        JoinDataset::type operator*() const
        {
            return _first_size > 0 ? *_iter1 : *_iter2;
        }

        iterator &operator++()
        {
            if(_first_size > 0)
            {
                --_first_size;
                ++_iter1;
            }
            else
            {
                ++_iter2;
            }

            return *this;
        }

    private:
        iter1_type _iter1;
        iter2_type _iter2;
        int        _first_size;
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
        return _dataset1.size() + _dataset2.size();
    }

private:
    T _dataset1;
    U _dataset2;
};

/** Helper function to create a @ref JoinDataset.
 *
 * @param[in] dataset1 First dataset.
 * @param[in] dataset2 Second dataset.
 *
 * @return A join dataset.
 */
template <typename T, typename U>
JoinDataset<T, U> concat(T &&dataset1, U &&dataset2)
{
    return JoinDataset<T, U>(std::forward<T>(dataset1), std::forward<U>(dataset2));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_JOIN */
