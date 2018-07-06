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
#ifndef ARM_COMPUTE_TEST_DATASET_LIST
#define ARM_COMPUTE_TEST_DATASET_LIST

#include "Dataset.h"
#include "utils/TypePrinter.h"

#include <initializer_list>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace dataset
{
/** Implementation of a dataset created from an initializer list. */
template <typename T>
class InitializerListDataset final : public NamedDataset
{
private:
    using data_const_iterator = typename std::vector<T>::const_iterator;

public:
    /** Construct dataset with given name and values from the container.
     *
     * @param[in] name Description of the values.
     * @param[in] list Values for the dataset.
     */
    InitializerListDataset(std::string name, std::initializer_list<T> &&list)
        : NamedDataset{ std::move(name) }, _data(std::forward<std::initializer_list<T>>(list))
    {
    }

    /** Allow instances of this class to be move constructed */
    InitializerListDataset(InitializerListDataset &&) = default;

    /** Type of the dataset. */
    using type = std::tuple<T>;

    /** Iterator for the dataset. */
    struct iterator
    {
        /** Construct an iterator for the dataset
         *
         * @param[in] name     Name of the dataset.
         * @param[in] iterator Iterator of the dataset values.
         */
        iterator(std::string name, data_const_iterator iterator)
            : _name{ name }, _iterator{ iterator }
        {
        }

        /** Get a description of the current value.
         *
         * @return a description.
         */
        std::string description() const
        {
            return _name + "=" + arm_compute::to_string(*_iterator);
        }

        /** Get the current value.
         *
         * @return the current value.
         */
        InitializerListDataset::type operator*() const
        {
            return std::make_tuple(*_iterator);
        }

        /** Increment the iterator.
         *
         * @return *this.
         */
        iterator &operator++()
        {
            ++_iterator;
            return *this;
        }

    private:
        std::string         _name;
        data_const_iterator _iterator;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(name(), _data.cbegin());
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return _data.size();
    }

private:
    std::vector<T> _data;
};

/** Helper function to create an @ref InitializerListDataset.
 *
 * @param[in] name Name of the dataset.
 * @param[in] list Initializer list.
 *
 * @return An initializer list dataset.
 */
template <typename T>
InitializerListDataset<T> make(std::string name, std::initializer_list<T> &&list)
{
    return InitializerListDataset<T>(std::move(name), std::forward<std::initializer_list<T>>(list));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_LIST */
