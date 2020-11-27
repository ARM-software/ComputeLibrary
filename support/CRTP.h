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
#ifndef ARM_COMPUTE_MISC_CRTP_H
#define ARM_COMPUTE_MISC_CRTP_H

namespace arm_compute
{
namespace misc
{
/** Curiously recurring template pattern Interface */
template <typename T, template <typename> class Type>
struct CRTP
{
public:
    /** Exact type */
    using ExactType = T;

protected:
    const T &impl() const
    {
        return static_cast<const T &>(*this);
    }
    T &impl()
    {
        return static_cast<T &>(*this);
    }

private:
    CRTP() = default;
    friend Type<T>;
};
} // namespace misc
} // namespace arm_compute
#endif /* ARM_COMPUTE_MISC_CRTP_H */
