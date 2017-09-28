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
/** Implementation of a dataset created from a range of values.
 *
 * The range is inclusive of the first value but exclusive of the last, i.e. [start, end).
 */
template <typename T>
class RangeDataset final : public NamedDataset
{
public:
    /** Construct dataset with given name and values in the specified range.
     *
     * @param[in] name  Description of the values.
     * @param[in] start Begin of the range.
     * @param[in] end   End of the range.
     * @param[in] step  Step size.
     */
    RangeDataset(std::string name, T start, T end, T step = 1)
        : NamedDataset{ std::move(name) }, _start{ start }, _end{ end }, _step{ step }
    {
    }

    RangeDataset(RangeDataset &&) = default;

    /** Type of the dataset. */
    using type = std::tuple<T>;

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(std::string name, T start, T step)
            : _name{ name }, _value{ start }, _step{ step }
        {
        }

        std::string description() const
        {
            using support::cpp11::to_string;
            return _name + "=" + to_string(_value);
        }

        RangeDataset::type operator*() const
        {
            return std::make_tuple(_value);
        }

        iterator &operator++()
        {
            _value += _step;
            return *this;
        }

    private:
        std::string _name;
        T           _value;
        T           _step;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(name(), _start, _step);
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return (_end - _start) / std::abs(_step);
    }

private:
    T _start;
    T _end;
    T _step;
};

/** Helper function to create a @ref RangeDataset.
 *
 * @param[in] name  Name of the dataset.
 * @param[in] start Begin of the range.
 * @param[in] end   End of the range.
 * @param[in] step  Step size.
 *
 * @return A range dataset.
 */
template <typename T>
RangeDataset<T> make(std::string name, T start, T end, T step = 1)
{
    return RangeDataset<T>(std::move(name), start, end, step);
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_RANGE */
