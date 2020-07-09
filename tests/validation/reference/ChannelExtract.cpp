/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "ChannelExtract.h"

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
template <typename T>
SimpleTensor<uint8_t> channel_extract(const TensorShape &shape, const std::vector<SimpleTensor<T>> &tensor_planes, Format format, Channel channel)
{
    // Find plane and channel index
    const unsigned int plane_idx   = plane_idx_from_channel(format, channel);
    const unsigned int channel_idx = channel_idx_from_format(format, channel);

    // Create dst and get src tensor
    SimpleTensor<T> src = tensor_planes[plane_idx];
    SimpleTensor<T> dst{ calculate_subsampled_shape(shape, format, channel), Format::U8 };

    // Single planar formats with subsampling require a double horizontal step
    const int step_x = ((Format::YUYV422 == format || Format::UYVY422 == format) && Channel::Y != channel) ? 2 : 1;
    const int width  = dst.shape().x();
    const int height = dst.shape().y();

    // Loop over each pixel and extract channel
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif /* _OPENMP */
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const Coordinates src_coord{ x * step_x, y };
            const Coordinates dst_coord{ x, y };

            const auto *src_pixel = reinterpret_cast<const T *>(src(src_coord));
            auto       *dst_pixel = reinterpret_cast<T *>(dst(dst_coord));

            dst_pixel[0] = src_pixel[channel_idx]; // NOLINT
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> channel_extract(const TensorShape &shape, const std::vector<SimpleTensor<uint8_t>> &tensor_planes, Format format, Channel channel);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
