/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MISC_CAST_H__
#define __ARM_COMPUTE_MISC_CAST_H__

#include "arm_compute/core/Error.h"

namespace arm_compute
{
namespace utils
{
namespace cast
{
/** Polymorphic cast between two types
 *
 * @warning Will throw an exception if cast cannot take place
 *
 * @tparam Target Target to cast type
 * @tparam Source Source from cast type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
template <typename Target, typename Source>
inline Target polymorphic_cast(Source *v)
{
    if(dynamic_cast<Target>(v) == nullptr)
    {
        ARM_COMPUTE_THROW(std::bad_cast());
    }
    return static_cast<Target>(v);
}

/** Polymorphic down cast between two types
 *
 * @warning Will assert if cannot take place
 *
 * @tparam Target Target to cast type
 * @tparam Source Source from cast type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
template <typename Target, typename Source>
inline Target polymorphic_downcast(Source *v)
{
    ARM_COMPUTE_ERROR_ON(dynamic_cast<Target>(v) != static_cast<Target>(v));
    return static_cast<Target>(v);
}

/** Polymorphic cast between two unique pointer types
 *
 * @warning Will throw an exception if cast cannot take place
 *
 * @tparam Target  Target to cast type
 * @tparam Source  Source from cast type
 * @tparam Deleter Deleter function type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
template <typename Target, typename Source, typename Deleter>
std::unique_ptr<Target, Deleter> polymorphic_cast_unique_ptr(std::unique_ptr<Source, Deleter> &&v)
{
    if(dynamic_cast<Target *>(v.get()) == nullptr)
    {
        ARM_COMPUTE_THROW(std::bad_cast());
    }
    auto r = static_cast<Target *>(v.release());
    return std::unique_ptr<Target, Deleter>(r, std::move(v.get_deleter()));
}

/** Polymorphic down cast between two unique pointer types
 *
 * @warning Will assert if cannot take place
 *
 * @tparam Target  Target to cast type
 * @tparam Source  Source from cast type
 * @tparam Deleter Deleter function type
 *
 * @param[in] v Value to cast
 *
 * @return The casted value
 */
template <typename Target, typename Source, typename Deleter>
std::unique_ptr<Target, Deleter> polymorphic_downcast_unique_ptr(std::unique_ptr<Source, Deleter> &&v)
{
    ARM_COMPUTE_ERROR_ON(dynamic_cast<Target *>(v.get()) != static_cast<Target *>(v.get()));
    auto r = static_cast<Target *>(v.release());
    return std::unique_ptr<Target, Deleter>(r, std::move(v.get_deleter()));
}
} // namespace cast
} // namespace utils
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_CAST_H__ */
