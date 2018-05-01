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
#include "ChannelShuffle.h"

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
// Refence implementation for channel shuffle taken from https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channel_shuffle_op.h
template <typename T>
SimpleTensor<T> channel_shuffle(const SimpleTensor<T> &src, int num_groups)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), src.num_channels(), src.fixed_point_position(), src.quantization_info() };

    const int M                 = src.shape()[0];
    const int N                 = src.shape()[1];
    const int num_channels      = src.shape()[2];
    const int batches           = src.shape()[3];
    const int MxN               = M * N;
    const int channels_in_group = num_channels / num_groups;

    const T *src_ref = src.data();
    T       *dst_ref = dst.data();

    for(int n = 0; n < batches; ++n)
    {
        for(int g = 0; g < num_groups; ++g)
        {
            // Gather the group g block (of size channels_in_group * MxN) from output channels
            // g + 0 * G, g + 1 * G, g + 2 * G, g + G * (K - 1) etc.
            const T *src_ptr = src_ref + g * MxN + n * num_channels * MxN;
            T       *dst_ptr = dst_ref + g * channels_in_group * MxN + n * num_channels * MxN;
            for(int i = 0; i < channels_in_group; ++i)
            {
                std::copy(src_ptr + i * num_groups * MxN,
                          src_ptr + (i * num_groups + 1) * MxN,
                          dst_ptr + i * MxN);
            }
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> channel_shuffle(const SimpleTensor<uint8_t> &src, int num_groups);
template SimpleTensor<uint16_t> channel_shuffle(const SimpleTensor<uint16_t> &src, int num_groups);
template SimpleTensor<uint32_t> channel_shuffle(const SimpleTensor<uint32_t> &src, int num_groups);
template SimpleTensor<half> channel_shuffle(const SimpleTensor<half> &src, int num_groups);
template SimpleTensor<float> channel_shuffle(const SimpleTensor<float> &src, int num_groups);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
