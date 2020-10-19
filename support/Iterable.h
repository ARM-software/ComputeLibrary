/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_MISC_ITERABLE_H
#define ARM_COMPUTE_MISC_ITERABLE_H

#include <iterator>

namespace arm_compute
{
namespace utils
{
namespace iterable
{
/** Reverse range iterable class
 *
 * @tparam T Type to create a reverse range on
 */
template <typename T>
class reverse_iterable
{
public:
    /** Default constructor
     *
     * @param[in] it Value to reverse iterate on
     */
    explicit reverse_iterable(T &it)
        : _it(it)
    {
    }

    /** Get beginning of iterator.
     *
     * @return beginning of iterator.
     */
    typename T::reverse_iterator begin()
    {
        return _it.rbegin();
    }

    /** Get end of iterator.
     *
     * @return end of iterator.
     */
    typename T::reverse_iterator end()
    {
        return _it.rend();
    }

    /** Get beginning of const iterator.
     *
     * @return beginning of const iterator.
     */
    typename T::const_reverse_iterator cbegin()
    {
        return _it.rbegin();
    }

    /** Get end of const iterator.
     *
     * @return end of const iterator.
     */
    typename T::const_reverse_iterator cend()
    {
        return _it.rend();
    }

private:
    T &_it;
};

/** Creates a reverse iterable for a given type
 *
 * @tparam T Type to create a reverse iterable on
 *
 * @param[in] val Iterable input
 *
 * @return Reverse iterable container
 */
template <typename T>
reverse_iterable<T> reverse_iterate(T &val)
{
    return reverse_iterable<T>(val);
}
} // namespace iterable
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_MISC_ITERABLE_H */
