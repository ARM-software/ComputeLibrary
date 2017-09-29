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
#ifndef __ARM_COMPUTE_TEST_RAWLUTACCESSOR_H__
#define __ARM_COMPUTE_TEST_RAWLUTACCESSOR_H__

#include "ILutAccessor.h"

#include <map>

namespace arm_compute
{
namespace test
{
/** Accessor implementation for std::map-lut objects. */
template <typename T>
class RawLutAccessor : public ILutAccessor<T>
{
public:
    /** Create an accessor for the given @p std::map.
     */
    RawLutAccessor(std::map<T, T> &lut)
        : _lut{ lut }
    {
    }

    RawLutAccessor(const RawLutAccessor &) = delete;
    RawLutAccessor &operator=(const RawLutAccessor &) = delete;
    RawLutAccessor(RawLutAccessor &&)                 = default;
    RawLutAccessor &operator=(RawLutAccessor &&) = default;

    int num_elements() const override
    {
        return _lut.size();
    }

    const T &operator[](T input_value) const override
    {
        return _lut[input_value];
    }

    T &operator[](T input_value) override
    {
        return _lut[input_value];
    }

private:
    std::map<T, T> &_lut;
};

} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_RAWLUTACCESSOR_H__ */
