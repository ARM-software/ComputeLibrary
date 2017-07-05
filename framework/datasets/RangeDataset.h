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
#ifndef ARM_COMPUTE_TEST_DATASET_RANGE
#define ARM_COMPUTE_TEST_DATASET_RANGE

#include "Dataset.h"
#include "support/ToolchainSupport.h"

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace dataset
{
/** Implementation of a dataset created from a range of values. */
template <typename T>
class RangeDataset final : public NamedDataset
{
public:
    /** Construct dataset with given name and values in the specified range.
     *
     * @param[in] name  Description of the values.
     * @param[in] first Iterator to the first value.
     * @param[in] last  Iterator behind the last value.
     */
    RangeDataset(std::string name, T &&first, T &&last)
        : NamedDataset{ std::move(name) }, _first{ std::forward<T>(first) }, _last{ std::forward<T>(last) }
    {
    }

    RangeDataset(RangeDataset &&) = default;

    /** Type of the dataset. */
    using type = std::tuple<typename std::iterator_traits<T>::value_type>;

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(std::string name, T iterator)
            : _name{ name }, _iterator{ iterator }
        {
        }

        std::string description() const
        {
            using support::cpp11::to_string;
            return _name + "=" + to_string(*_iterator);
        }

        RangeDataset::type operator*() const
        {
            return std::make_tuple(*_iterator);
        }

        iterator &operator++()
        {
            ++_iterator;
            return *this;
        }

    private:
        std::string _name;
        T           _iterator;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(name(), _first);
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return std::distance(_first, _last);
    }

private:
    T _first;
    T _last;
};

/** Helper function to create a @ref RangeDataset.
 *
 * @param[in] name  Name of the dataset.
 * @param[in] first Iterator to the first value.
 * @param[in] last  Iterator behind the last value.
 *
 * @return A range dataset.
 */
template <typename T>
RangeDataset<T> make(std::string name, T &&first, T &&last)
{
    return RangeDataset<T>(std::move(name), std::forward<T>(first), std::forward<T>(last));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_RANGE */
