/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_SUPPORT_MEMORYSUPPORT
#define ARM_COMPUTE_SUPPORT_MEMORYSUPPORT

#include <memory>

namespace arm_compute
{
namespace support
{
namespace cpp11
{
// std::align is missing in GCC 4.9
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
inline void *align(std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space)
{
    std::uintptr_t pn      = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t aligned = (pn + alignment - 1) & -alignment;
    std::size_t    padding = aligned - pn;
    if(space < size + padding)
    {
        return nullptr;
    }

    space -= padding;

    return ptr = reinterpret_cast<void *>(aligned);
}
} //namespace cpp11
namespace cpp14
{
/** make_unique is missing in CPP11. Re-implement it according to the standard proposal. */

/**<Template for single object */
template <class T>
struct _Unique_if
{
    typedef std::unique_ptr<T> _Single_object; /**< Single object type */
};

/** Template for array */
template <class T>
struct _Unique_if<T[]>
{
    typedef std::unique_ptr<T[]> _Unknown_bound; /**< Array type */
};

/** Template for array with known bounds (to throw an error).
 *
 * @note this is intended to never be hit.
 */
template <class T, size_t N>
struct _Unique_if<T[N]>
{
    typedef void _Known_bound; /**< Should never be used */
};

/** Construct a single object and return a unique pointer to it.
 *
 * @param[in] args Constructor arguments.
 *
 * @return a unique pointer to the new object.
 */
template <class T, class... Args>
typename _Unique_if<T>::_Single_object
make_unique(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/** Construct an array of objects and return a unique pointer to it.
 *
 * @param[in] n Array size
 *
 * @return a unique pointer to the new array.
 */
template <class T>
typename _Unique_if<T>::_Unknown_bound
make_unique(size_t n)
{
    typedef typename std::remove_extent<T>::type U;
    return std::unique_ptr<T>(new U[n]());
}

/** It is invalid to attempt to make_unique an array with known bounds. */
template <class T, class... Args>
typename _Unique_if<T>::_Known_bound
make_unique(Args &&...) = delete;
} // namespace cpp14
} // namespace support
} // namespace arm_compute
#endif /* ARM_COMPUTE_SUPPORT_MEMORYSUPPORT */
