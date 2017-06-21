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
#ifndef __ARM_COMPUTE_TEST_PADDING_CALCULATOR_H__
#define __ARM_COMPUTE_TEST_PADDING_CALCULATOR_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
/** Calculate required padding. */
class PaddingCalculator final
{
public:
    /** Construct calculator with size of tensor's dimension and step size.
     *
     * @param[in] size               Number of elements available.
     * @param[in] processed_elements Number of elements processed per iteration.
     */
    PaddingCalculator(int size, int processed_elements)
        : _size{ size }, _num_processed_elements{ processed_elements }, _num_accessed_elements{ processed_elements }
    {
    }

    /** Set border mode.
     *
     * @param[in] mode Border mode.
     */
    void set_border_mode(BorderMode mode);

    /** Set border size.
     *
     * @param[in] size Border size in elements.
     */
    void set_border_size(int size);

    /** Set offset of the access relative to the current position.
     *
     * @param[in] offset Offset of the access.
     */
    void set_access_offset(int offset);

    /** Set number of accessed elements.
     *
     * @param[in] elements Number of accessed elements per iteration.
     */
    void set_accessed_elements(int elements);

    /** Compute the required padding.
     *
     * @return Required padding in number of elements.
     */
    int required_padding() const;

private:
    int        _size;
    int        _num_processed_elements;
    int        _num_accessed_elements;
    BorderMode _mode{ BorderMode::UNDEFINED };
    int        _border_size{ 0 };
    int        _offset{ 0 };
};

inline void PaddingCalculator::set_border_mode(BorderMode mode)
{
    _mode = mode;
}

inline void PaddingCalculator::set_border_size(int size)
{
    _border_size = size;
}

inline void PaddingCalculator::set_access_offset(int offset)
{
    _offset = offset;
}

inline void PaddingCalculator::set_accessed_elements(int elements)
{
    _num_accessed_elements = elements;
}

inline int PaddingCalculator::required_padding() const
{
    if(_mode == BorderMode::UNDEFINED)
    {
        return (((_size - _border_size + _num_processed_elements - 1) / _num_processed_elements) - 1) * _num_processed_elements + _num_accessed_elements - _size + _border_size + _offset;
    }
    else
    {
        return (((_size + _num_processed_elements - 1) / _num_processed_elements) - 1) * _num_processed_elements + _num_accessed_elements - _size + _offset;
    }
}
} // namespace test
} // namespace arm_compute
#endif
