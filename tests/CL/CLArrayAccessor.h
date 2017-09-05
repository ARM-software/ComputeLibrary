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
#ifndef __ARM_COMPUTE_TEST_CLARRAYACCESSOR_H__
#define __ARM_COMPUTE_TEST_CLARRAYACCESSOR_H__

#include "arm_compute/runtime/CL/CLArray.h"
#include "tests/IArrayAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref CLArray objects. */
template <typename T>
class CLArrayAccessor : public IArrayAccessor<T>
{
public:
    /** Create an accessor for the given @p array.
     *
     * @param[in, out] array To be accessed array.
     *
     * @note The CL memory is mapped by the constructor.
     *
     */
    CLArrayAccessor(CLArray<T> &array)
        : _array{ array }
    {
        _array.map();
    }

    CLArrayAccessor(const CLArrayAccessor &) = delete;
    CLArrayAccessor &operator=(const CLArrayAccessor &) = delete;
    CLArrayAccessor(CLArrayAccessor &&)                 = default;
    CLArrayAccessor &operator=(CLArrayAccessor &&) = default;

    /** Destructor that unmaps the CL memory. */
    ~CLArrayAccessor()
    {
        _array.unmap();
    }

    size_t num_values() const override
    {
        return _array.num_values();
    }

    T *buffer() override
    {
        return _array.buffer();
    }

    void resize(size_t num) override
    {
        _array.resize(num);
    }

    T &at(size_t index) const override
    {
        ARM_COMPUTE_ERROR_ON(_array.buffer() == nullptr);
        ARM_COMPUTE_ERROR_ON(index >= num_values());
        return _array.buffer()[index];
    }

private:
    CLArray<T> &_array;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_CLARRAYACCESSOR_H__ */
