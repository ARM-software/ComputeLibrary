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
#ifndef ARM_COMPUTE_TEST_DATASET_CONTAINER
#define ARM_COMPUTE_TEST_DATASET_CONTAINER

#include "Dataset.h"
#include "support/ToolchainSupport.h"

#include <string>
#include <tuple>
#include <type_traits>
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
/** Base case. Nothing is a container. */
template <typename T>
struct is_container : public std::false_type
{
};

/** Vector is considered a container. */
template <typename V, typename A>
struct is_container<std::vector<V, A>> : public std::true_type
{
};

/** Implementation of a dataset created from a container. */
template <typename T>
class ContainerDataset : public NamedDataset
{
private:
    using container_value_type     = typename T::value_type;
    using container_const_iterator = typename T::const_iterator;

public:
    /** Construct dataset with given name and values from the container.
     *
     * @param[in] name      Description of the values.
     * @param[in] container Values for the dataset.
     */
    ContainerDataset(std::string name, T &&container)
        : NamedDataset{ std::move(name) }, _container(std::forward<T>(container))
    {
    }

    ContainerDataset(ContainerDataset &&) = default;

    /** Type of the dataset. */
    using type = std::tuple<container_value_type>;

    /** Iterator for the dataset. */
    struct iterator
    {
        iterator(std::string name, container_const_iterator iterator)
            : _name{ name }, _iterator{ iterator }
        {
        }

        std::string description() const
        {
            using support::cpp11::to_string;
            return _name + "=" + to_string(*_iterator);
        }

        ContainerDataset::type operator*() const
        {
            return std::make_tuple(*_iterator);
        }

        iterator &operator++()
        {
            ++_iterator;
            return *this;
        }

    private:
        std::string              _name;
        container_const_iterator _iterator;
    };

    /** Iterator pointing at the begin of the dataset.
     *
     * @return Iterator for the dataset.
     */
    iterator begin() const
    {
        return iterator(name(), _container.cbegin());
    }

    /** Size of the dataset.
     *
     * @return Number of values in the dataset.
     */
    int size() const
    {
        return _container.size();
    }

private:
    T _container;
};

/** Helper function to create a @ref ContainerDataset.
 *
 * @param[in] name   Name of the dataset.
 * @param[in] values Container.
 *
 * @return A container dataset.
 */
template <typename T>
typename std::enable_if<is_container<T>::value, ContainerDataset<T>>::type make(std::string name, T &&values)
{
    return ContainerDataset<T>(std::move(name), std::forward<T>(values));
}
} // namespace dataset
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DATASET_CONTAINER */
