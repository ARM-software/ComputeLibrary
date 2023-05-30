/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef CKW_INCLUDE_CKW_SCALARVALUE_H
#define CKW_INCLUDE_CKW_SCALARVALUE_H

#include "ckw/Error.h"

#include <cstdint>

namespace ckw
{

/** The scalar value known at compile-time. */
class ScalarValue
{
public:
    /** Initialize a new instance of @ref ScalarValue class with integer value 0. */
    ScalarValue()
    {
        _type      = Type::INT;
        _value.i64 = 0;
    }

    /** Initialize a new instance of @ref ScalarValue class with the specified value. */
    template <typename T>
    ScalarValue(T value)
    {
        set(value);
    }

    /** Set the value. */
    template <typename T>
    void set(T value)
    {
        CKW_ASSERT(::std::is_integral<T>::value || ::std::is_floating_point<T>::value);
        CKW_ASSERT(sizeof(T) <= 8);

        _size = sizeof(T);

        if(::std::is_integral<T>::value)
        {
            if(::std::is_signed<T>::value)
            {
                _type      = Type::INT;
                _value.i64 = value;
            }
            else
            {
                _type      = Type::UINT;
                _value.u64 = value;
            }
        }
        else
        {
            _type      = Type::FLOAT;
            _value.f64 = value;
        }
    }

    /** Get the value.
     *
     * The caller must make sure that what has been stored in the object must fit
     * the output data type without data corruption or loss of accuracy.
     */
    template <typename T>
    T get() const
    {
        CKW_ASSERT(::std::is_integral<T>::value || ::std::is_floating_point<T>::value);
        CKW_ASSERT(sizeof(T) >= _size);

        if(::std::is_integral<T>::value)
        {
            if(::std::is_signed<T>::value)
            {
                CKW_ASSERT(_type == Type::INT || _type == Type::UINT);
                CKW_ASSERT_IF(_type == Type::UINT, sizeof(T) > _size);

                return _value.i64;
            }
            else
            {
                CKW_ASSERT(_type == Type::INT);

                return _value.u64;
            }
        }
        else
        {
            return _value.f64;
        }
    }

private:
    union Value
    {
        int64_t  i64;
        uint64_t u64;
        double   f64;
    };

    enum class Type : int32_t
    {
        UINT,
        INT,
        FLOAT,
    };

    Value    _value{};
    Type     _type{};
    uint32_t _size{};
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_SCALARVALUE_H
