/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_IARRAY_H__
#define __ARM_COMPUTE_IARRAY_H__

#include "arm_compute/core/Error.h"
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
struct KeyPoint;
struct Coordinates2D;
struct DetectionWindow;
class Size2D;

/** Array of type T */
template <class T>
class IArray
{
public:
    /** Default constructor */
    IArray()
        : _num_values(0), _max_size(0) {};
    /** Constructor: initializes an array which can contain up to max_num_points values
     *
     * @param[in] max_num_values Maximum number of values the array will be able to stored
     */
    IArray(size_t max_num_values)
        : _num_values(0), _max_size(max_num_values)
    {
    }
    /** Maximum number of values which can be stored in this array
     *
     * @return Maximum number of values
     */
    size_t max_num_values() const
    {
        return _max_size;
    }
    /** Default virtual destructor */
    virtual ~IArray() = default;
    /** Number of values currently stored in the array
     *
     * @return Number of values currently stored in the array or max_num_values + 1 if the array is overflowed.
     */
    size_t num_values() const
    {
        return _num_values;
    }
    /** Append the passed argument to the end of the array if there is room.
     *
     * @param[in] val Value to add to the array.
     *
     * @return True if the point was successfully added to the array. False if the array is full and the point couldn't be added.
     */
    bool push_back(const T &val)
    {
        ARM_COMPUTE_ERROR_ON(0 == _max_size);
        if(_num_values >= max_num_values())
        {
            _num_values = max_num_values() + 1;
            return false;
        }
        at(_num_values) = val;
        _num_values++;
        return true;
    }
    /** Clear all the points from the array. */
    void clear()
    {
        _num_values = 0;
    }
    /** Did we lose some values because the array is too small?
     *
     * @return True if we tried to add a value using push_back() but there wasn't any room left to store it.
     * False if all the values were successfully added to the array.
     */
    bool overflow() const
    {
        return _num_values > max_num_values();
    }
    /** Pointer to the first element of the array
     *
     * Other elements of the array can be accessed using buffer()[idx] for 0 <= idx < num_poins().
     *
     * @return A pointer to the first element of the array
     */
    virtual T *buffer() const = 0;
    /** Reference to the element of the array located at the given index
     *
     * @param[in] index Index of the element
     *
     * @return A reference to the element of the array located at the given index.
     */
    virtual T &at(size_t index) const
    {
        ARM_COMPUTE_ERROR_ON(buffer() == nullptr);
        ARM_COMPUTE_ERROR_ON(index >= max_num_values());
        return buffer()[index];
    }
    /** Resizes the array to contain "num" elements. If "num" is smaller than the maximum array size, the content is reduced to its first "num" elements,
     *  "num" elements can't be bigger than the maximum number of values which can be stored in this array.
     *
     * @param[in] num The new array size in number of elements
     */
    void resize(size_t num)
    {
        ARM_COMPUTE_ERROR_ON(num > max_num_values());
        _num_values = num;
    };

private:
    size_t _num_values;
    size_t _max_size;
};
/** Interface for Array of Key Points. */
using IKeyPointArray = IArray<KeyPoint>;
/** Interface for Array of 2D Coordinates. */
using ICoordinates2DArray = IArray<Coordinates2D>;
/** Interface for Array of Detection Windows. */
using IDetectionWindowArray = IArray<DetectionWindow>;
/** Interface for Array of 2D Sizes. */
using ISize2DArray = IArray<Size2D>;
/** Interface for Array of uint8s. */
using IUInt8Array = IArray<uint8_t>;
/** Interface for Array of uint16s. */
using IUInt16Array = IArray<uint16_t>;
/** Interface for Array of uint32s. */
using IUInt32Array = IArray<uint32_t>;
/** Interface for Array of int16s. */
using IInt16Array = IArray<int16_t>;
/** Interface for Array of int32s. */
using IInt32Array = IArray<int32_t>;
/** Interface for Array of floats. */
using IFloatArray = IArray<float>;
}
#endif /* __ARM_COMPUTE_IARRAY_H__ */
