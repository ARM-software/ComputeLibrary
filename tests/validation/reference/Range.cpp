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
#include "arm_compute/core/Types.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
template <typename T>
void generate_range(SimpleTensor<T> &dst, float start, const size_t num_of_elements, float step)
{
    float val = start;
    for(size_t index = 0; index < num_of_elements; index++)
    {
        dst[index] = static_cast<T>(val);
        val += step;
    }
}
} // namespace

template <typename T>
SimpleTensor<T> range(SimpleTensor<T> &dst, float start, const size_t num_of_elements, float step)
{
    generate_range(dst, start, num_of_elements, step);
    return dst;
}

template <>
SimpleTensor<uint8_t> range(SimpleTensor<uint8_t> &dst, float start, const size_t num_of_elements, float step)
{
    if(dst.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> dst_tmp{ dst.shape(), DataType::F32, 1 };
        generate_range(dst_tmp, start, num_of_elements, step);
        return convert_to_asymmetric(dst_tmp, dst.quantization_info());
    }
    generate_range(dst, start, num_of_elements, step);
    return dst;
}
template SimpleTensor<float> range(SimpleTensor<float> &dst, float start, const size_t num_of_elements, float step);
template SimpleTensor<half> range(SimpleTensor<half> &dst, float start, const size_t num_of_elements, float step);
template SimpleTensor<int8_t> range(SimpleTensor<int8_t> &dst, float start, const size_t num_of_elements, float step);
template SimpleTensor<uint16_t> range(SimpleTensor<uint16_t> &dst, float start, const size_t num_of_elements, float step);
template SimpleTensor<int16_t> range(SimpleTensor<int16_t> &dst, float start, const size_t num_of_elements, float step);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
