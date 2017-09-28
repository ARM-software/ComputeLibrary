/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_ARRAY_H__
#define __ARM_COMPUTE_ARRAY_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
/** Basic implementation of the IArray interface which allocates a static number of T values  */
template <class T>
class Array : public IArray<T>
{
public:
    /** Default constructor: empty array */
    Array()
        : IArray<T>(0), _values(nullptr)
    {
    }
    /** Constructor: initializes an array which can contain up to max_num_points values
     *
     * @param[in] max_num_values Maximum number of values the array will be able to stored
     */
    Array(size_t max_num_values)
        : IArray<T>(max_num_values), _values(arm_compute::support::cpp14::make_unique<T[]>(max_num_values))
    {
    }

    // Inherited methods overridden:
    T *buffer() const override
    {
        return _values.get();
    }

private:
    std::unique_ptr<T[]> _values;
};

using KeyPointArray        = Array<KeyPoint>;
using Coordinates2DArray   = Array<Coordinates2D>;
using DetectionWindowArray = Array<DetectionWindow>;
using Size2DArray          = Array<Size2D>;
using UInt8Array           = Array<uint8_t>;
using UInt16Array          = Array<uint16_t>;
using UInt32Array          = Array<uint32_t>;
using Int16Array           = Array<int16_t>;
using Int32Array           = Array<int32_t>;
using FloatArray           = Array<float>;
}
#endif /* __ARM_COMPUTE_ARRAY_H__ */
