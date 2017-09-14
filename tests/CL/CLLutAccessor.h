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
#ifndef __ARM_COMPUTE_TEST_CL_CLLUTACCESSOR_H__
#define __ARM_COMPUTE_TEST_CL_CLLUTACCESSOR_H__

#include "tests/ILutAccessor.h"

#include "arm_compute/runtime/CL/CLLut.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref CLLut objects. */
template <typename T>
class CLLutAccessor : public ILutAccessor<T>
{
public:
    /** Create an accessor for the given @p CLLut.
     */
    CLLutAccessor(CLLut &lut)
        : _lut{ lut }
    {
        _lut.map(true);
    }
    ~CLLutAccessor()
    {
        _lut.unmap();
    }

    CLLutAccessor(const CLLutAccessor &) = delete;
    CLLutAccessor &operator=(const CLLutAccessor &) = delete;
    CLLutAccessor(CLLutAccessor &&)                 = default;
    CLLutAccessor &operator=(CLLutAccessor &&) = default;

    int num_elements() const override
    {
        return _lut.num_elements();
    }

    const T &operator[](T input_value) const override
    {
        auto    lut        = reinterpret_cast<T *>(_lut.buffer());
        int32_t real_index = _lut.index_offset() + static_cast<int32_t>(input_value);

        if(0 <= real_index && real_index < num_elements())
        {
            return lut[real_index];
        }
        ARM_COMPUTE_ERROR("Error index not in range.");
    }

    T &operator[](T input_value) override
    {
        auto    lut        = reinterpret_cast<T *>(_lut.buffer());
        int32_t real_index = _lut.index_offset() + static_cast<int32_t>(input_value);

        if(0 <= real_index && real_index < num_elements())
        {
            return lut[real_index];
        }
        ARM_COMPUTE_ERROR("Error index not in range.");
    }

private:
    CLLut &_lut;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_CL_CLLUTACCESSOR_H__ */
