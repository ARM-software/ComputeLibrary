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
#ifndef ARM_COMPUTE_TEST_DATASET_SINGLETON
#define ARM_COMPUTE_TEST_DATASET_SINGLETON

#include "ContainerDataset.h"
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
/** Implementation of a dataset holding a single value. */
template <typename T>
class SingletonDataset : public NamedDataset
{
public:
    /** Construct dataset with given name and value.
     *
     * @param[in] name  Description of the value.
     * @param[in] value Value for the dataset.
     */
    SingletonDataset(std::string name, T &&value)
        : NamedDataset{ std::move(name) }, _value{ std::forward<T>(value) }
    {
    }

    SingletonDataset(SingletonDataset &&) = default;

    /** Type of the dataset. */
    using type = std::tuple<T>;

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(std::string name, const T *value)
            : _name{ name }, _value{ value }
        {
        }

        ~iterator() = default;

        iterator(const iterator &) = default;
        iterator &operator=(const iterator &) = default;
        iterator(iterator &&)                 = default;
        iterator &operator=(iterator &&) = default;

        std::string description() const
        {
            using support::cpp11::to_string;
            return _name + "=" + to_string(*_value);
        }

        SingletonDataset::type operator*() const
        {
            return std::make_tuple(*_value);
        }

        iterator &operator++()
        {
            return *this;
        }

    private:
        std::string _name;
        const T    *_value;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(name(), &_value);
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return 1;
    }

private:
    T _value;
};

/** Helper function to create a @ref SingletonDataset.
 *
 * @param[in] name  Name of the dataset.
 * @param[in] value Value.
 *
 * @return A singleton dataset.
 */
template <typename T>
typename std::enable_if < !is_container<T>::value, SingletonDataset<T >>::type make(std::string name, T &&value)
{
    return SingletonDataset<T>(std::move(name), std::forward<T>(value));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_SINGLETON */
