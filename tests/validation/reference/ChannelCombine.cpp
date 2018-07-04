/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "ChannelCombine.h"

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
inline std::vector<SimpleTensor<T>> create_image_planes(const TensorShape &shape, Format format)
{
    TensorShape image_shape = adjust_odd_shape(shape, format);

    std::vector<SimpleTensor<T>> image_planes;

    switch(format)
    {
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
        {
            image_planes.emplace_back(image_shape, format);
            break;
        }
        case Format::NV12:
        case Format::NV21:
        {
            TensorShape shape_uv88 = calculate_subsampled_shape(image_shape, Format::UV88);

            image_planes.emplace_back(image_shape, Format::U8);
            image_planes.emplace_back(shape_uv88, Format::UV88);
            break;
        }
        case Format::IYUV:
        {
            TensorShape shape_sub2 = calculate_subsampled_shape(image_shape, Format::IYUV);

            image_planes.emplace_back(image_shape, Format::U8);
            image_planes.emplace_back(shape_sub2, Format::U8);
            image_planes.emplace_back(shape_sub2, Format::U8);
            break;
        }
        case Format::YUV444:
        {
            image_planes.emplace_back(image_shape, Format::U8);
            image_planes.emplace_back(image_shape, Format::U8);
            image_planes.emplace_back(image_shape, Format::U8);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    return image_planes;
}
} // namespace

template <typename T>
std::vector<SimpleTensor<T>> channel_combine(const TensorShape &shape, const std::vector<SimpleTensor<T>> &image_planes, Format format)
{
    std::vector<SimpleTensor<T>> dst = create_image_planes<T>(shape, format);

    for(unsigned int plane_idx = 0; plane_idx < dst.size(); ++plane_idx)
    {
        SimpleTensor<T> &dst_tensor = dst[plane_idx];

        for(int element_idx = 0; element_idx < dst_tensor.num_elements(); ++element_idx)
        {
            Coordinates coord = index2coord(dst_tensor.shape(), element_idx);

            switch(format)
            {
                case Format::RGB888:
                case Format::RGBA8888:
                {
                    // Copy R/G/B or A channel
                    for(int channel_idx = 0; channel_idx < dst_tensor.num_channels(); ++channel_idx)
                    {
                        const T &src_value = reinterpret_cast<const T *>(image_planes[channel_idx](coord))[0];
                        T       &dst_value = reinterpret_cast<T *>(dst_tensor(coord))[channel_idx];

                        dst_value = src_value;
                    }
                    break;
                }
                case Format::YUYV422:
                case Format::UYVY422:
                {
                    // Find coordinates of the sub-sampled pixel
                    const Coordinates coord_hori(coord.x() / 2, coord.y());

                    const T &src0 = reinterpret_cast<const T *>(image_planes[0](coord))[0];
                    const T &src1 = reinterpret_cast<const T *>(image_planes[1](coord_hori))[0];

                    const int shift = (Format::YUYV422 == format) ? 1 : 0;
                    T        &dst0  = reinterpret_cast<T *>(dst_tensor(coord))[1 - shift];
                    T        &dst1  = reinterpret_cast<T *>(dst_tensor(coord))[0 + shift];

                    dst0 = src0;
                    dst1 = src1;

                    Coordinates coord2 = index2coord(dst_tensor.shape(), ++element_idx);

                    const T &src2 = reinterpret_cast<const T *>(image_planes[0](coord2))[0];
                    const T &src3 = reinterpret_cast<const T *>(image_planes[2](coord_hori))[0];

                    T &dst2 = reinterpret_cast<T *>(dst_tensor(coord2))[1 - shift];
                    T &dst3 = reinterpret_cast<T *>(dst_tensor(coord2))[0 + shift];

                    dst2 = src2;
                    dst3 = src3;

                    break;
                }
                case Format::NV12:
                case Format::NV21:
                {
                    if(0U == plane_idx)
                    {
                        // Get and combine Y channel from plane0 of destination multi-image
                        dst_tensor[element_idx] = image_planes[0][element_idx];
                    }
                    else
                    {
                        const int shift = (Format::NV12 == format) ? 0 : 1;

                        // Get U channel from plane1 and V channel from plane2 of the source
                        const T &src_u0 = reinterpret_cast<const T *>(image_planes[1](coord))[0];
                        const T &src_v0 = reinterpret_cast<const T *>(image_planes[2](coord))[0];

                        // Get U and V channel from plane1 of destination multi-image
                        T &dst_u0 = reinterpret_cast<T *>(dst_tensor(coord))[0 + shift];
                        T &dst_v0 = reinterpret_cast<T *>(dst_tensor(coord))[1 - shift];

                        // Combine channel U and V
                        dst_u0 = src_u0;
                        dst_v0 = src_v0;
                    }

                    break;
                }
                case Format::IYUV:
                case Format::YUV444:
                {
                    // Get Y/U/V element
                    const T &src = reinterpret_cast<const T *>(image_planes[plane_idx](coord))[0];
                    T       &dst = reinterpret_cast<T *>(dst_tensor(coord))[0];

                    // Copy Y/U/V plane
                    dst = src;

                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
        }
    }

    return dst;
}

template std::vector<SimpleTensor<uint8_t>> channel_combine(const TensorShape &shape, const std::vector<SimpleTensor<uint8_t>> &image_planes, Format format);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
