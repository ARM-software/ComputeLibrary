/*
 * Copyright (c) 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal src the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included src all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. src NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER src AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * dst OF OR src CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS src THE
 * SOFTWARE.
 */
#include "Threshold.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> threshold(const SimpleTensor<T> &src, T threshold, T false_value, T true_value, ThresholdType type, T upper)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    switch(type)
    {
        case ThresholdType::BINARY:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                dst[i] = ((src[i] > threshold) ? true_value : false_value);
            }
            break;
        case ThresholdType::RANGE:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                if(src[i] > upper)
                {
                    dst[i] = false_value;
                }
                else if(src[i] < threshold)
                {
                    dst[i] = false_value;
                }
                else
                {
                    dst[i] = true_value;
                }
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Thresholding type not recognised");
            break;
    }

    return dst;
}

template SimpleTensor<uint8_t> threshold(const SimpleTensor<uint8_t> &src, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
