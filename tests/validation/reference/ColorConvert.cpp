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
#include "ColorConvert.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ColorConvertHelper.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<uint8_t> color_convert(const TensorShape &shape, const std::vector<SimpleTensor<T>> &tensor_planes, Format src_format, Format dst_format)
{
    // Create dst and get src tensor
    SimpleTensor<T> src = tensor_planes[0];
    SimpleTensor<T> dst(shape, dst_format);

    switch(src_format)
    {
        case Format::RGB888:
        {
            switch(dst_format)
            {
                case Format::RGBA8888:
                    colorconvert_helper::detail::colorconvert_rgb_to_rgbx(src, dst);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not Supported");
                    break;
            }
            break;
        }
        case Format::RGBA8888:
        {
            switch(dst_format)
            {
                case Format::RGB888:
                    colorconvert_helper::detail::colorconvert_rgbx_to_rgb(src, dst);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not Supported");
                    break;
            }
            break;
        }
        case Format::UYVY422:
        case Format::YUYV422:
        {
            switch(dst_format)
            {
                case Format::RGB888:
                case Format::RGBA8888:
                    colorconvert_helper::detail::colorconvert_yuyv_to_rgb(src, src_format, dst);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not Supported");
                    break;
            }
            break;
        }
        case Format::IYUV:
        {
            switch(dst_format)
            {
                case Format::RGB888:
                case Format::RGBA8888:
                    colorconvert_helper::detail::colorconvert_iyuv_to_rgb(shape, tensor_planes, dst);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not Supported");
                    break;
            }
            break;
        }
        case Format::NV12:
        case Format::NV21:
        {
            switch(dst_format)
            {
                case Format::RGB888:
                case Format::RGBA8888:
                    colorconvert_helper::detail::colorconvert_nv12_to_rgb(shape, src_format, tensor_planes, dst);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not Supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    return dst;
}

template SimpleTensor<uint8_t> color_convert(const TensorShape &shape, const std::vector<SimpleTensor<uint8_t>> &tensor_planes, Format src_format, Format dst_format);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
