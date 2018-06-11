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
 *asymm_int_mult
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, asymm_int_multDAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_TEST_VALIDATION_COLOR_CONVERT_H__
#define __ARM_COMPUTE_TEST_VALIDATION_COLOR_CONVERT_H__

#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace colorconvert_helper
{
namespace detail
{
constexpr float red_coef_bt709    = 1.5748F;
constexpr float green_coef_bt709  = -0.1873f;
constexpr float green_coef2_bt709 = -0.4681f;
constexpr float blue_coef_bt709   = 1.8556f;

template <typename T>
inline void yuyv_to_rgb_calculation(const SimpleTensor<T> yvec, const SimpleTensor<T> vvec, const SimpleTensor<T> yyvec, const SimpleTensor<T> uvec, SimpleTensor<T> &dst)
{
    const int dst_width  = dst.shape().x();
    const int dst_height = dst.shape().y();

    for(int y = 0; y < dst_height; ++y)
    {
        int x_coord = 0;
        for(int x = 0; x < dst_width; x += 2, ++x_coord)
        {
            Coordinates dst_coord{ x, y };
            auto       *dst_pixel = reinterpret_cast<T *>(dst(dst_coord));
            float       result    = 0.f;

            T           border_value(0);
            const int   yvec_val  = validation::tensor_elem_at(yvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int   vvec_val  = validation::tensor_elem_at(vvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int   yyvec_val = validation::tensor_elem_at(yyvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int   uvec_val  = validation::tensor_elem_at(uvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const float red       = (vvec_val - 128) * red_coef_bt709;
            const float green     = (uvec_val - 128) * green_coef_bt709 + (vvec_val - 128) * green_coef2_bt709;
            const float blue      = (uvec_val - 128) * blue_coef_bt709;

            for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
            {
                if(channel_idx == 0)
                {
                    // Channel 'R'
                    result = yvec_val + red;
                }
                else if(channel_idx == 1)
                {
                    // Channel 'G'
                    result = yvec_val + green;
                }
                else if(channel_idx == 2)
                {
                    // Channel 'B'
                    result = yvec_val + blue;
                }
                else
                {
                    // Channel 'A'
                    result = 255;
                }

                if(result < 0)
                {
                    result = 0;
                }
                else if(result > 255)
                {
                    result = 255;
                }
                dst_pixel[channel_idx] = result;
            }

            dst_coord.set(0, x + 1);
            dst_pixel = reinterpret_cast<T *>(dst(dst_coord));
            for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
            {
                if(channel_idx == 0)
                {
                    // Channel 'R'
                    result = yyvec_val + red;
                }
                else if(channel_idx == 1)
                {
                    // Channel 'G'
                    result = yyvec_val + green;
                }
                else if(channel_idx == 2)
                {
                    // Channel 'B'
                    result = yyvec_val + blue;
                }
                else
                {
                    // Channel 'A'
                    result = 255;
                }

                if(result < 0)
                {
                    result = 0;
                }
                else if(result > 255)
                {
                    result = 255;
                }
                dst_pixel[channel_idx] = result;
            }
        }
    }
}

template <typename T>
inline void colorconvert_rgb_to_rgbx(const SimpleTensor<T> src, SimpleTensor<T> &dst)
{
    for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
    {
        const int width  = dst.shape().x();
        const int height = dst.shape().y();

        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                const Coordinates src_coord{ x, y };
                const Coordinates dst_coord{ x, y };

                const auto *src_pixel = reinterpret_cast<const T *>(src(src_coord));
                auto       *dst_pixel = reinterpret_cast<T *>(dst(dst_coord));
                if(channel_idx == 3)
                {
                    dst_pixel[channel_idx] = 255;
                    continue;
                }

                dst_pixel[channel_idx] = src_pixel[channel_idx];
            }
        }
    }
}

template <typename T>
inline void colorconvert_rgbx_to_rgb(const SimpleTensor<T> src, SimpleTensor<T> &dst)
{
    for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
    {
        const int width  = dst.shape().x();
        const int height = dst.shape().y();

        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                const Coordinates src_coord{ x, y };
                const Coordinates dst_coord{ x, y };

                const auto *src_pixel = reinterpret_cast<const T *>(src(src_coord));
                auto       *dst_pixel = reinterpret_cast<T *>(dst(dst_coord));

                dst_pixel[channel_idx] = src_pixel[channel_idx];
            }
        }
    }
}

template <typename T>
inline void colorconvert_yuyv_to_rgb(const SimpleTensor<T> src, const Format format, SimpleTensor<T> &dst)
{
    SimpleTensor<T> yvec(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    SimpleTensor<T> uvec(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    SimpleTensor<T> yyvec(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    SimpleTensor<T> vvec(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);

    const int step_x = (Format::YUYV422 == format || Format::UYVY422 == format) ? 2 : 1;
    const int offset = (Format::YUYV422 == format) ? 0 : 1;

    Coordinates elem_coord{ 0, 0 };
    const int   width  = yvec.shape().x();
    const int   height = yvec.shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const Coordinates src_coord{ x * step_x, y };
            const auto       *src_pixel   = reinterpret_cast<const T *>(src(src_coord));
            auto             *yvec_pixel  = reinterpret_cast<T *>(yvec(elem_coord));
            auto             *uvec_pixel  = reinterpret_cast<T *>(uvec(elem_coord));
            auto             *yyvec_pixel = reinterpret_cast<T *>(yyvec(elem_coord));
            auto             *vvec_pixel  = reinterpret_cast<T *>(vvec(elem_coord));
            yvec_pixel[x]                 = src_pixel[0 + offset];
            uvec_pixel[x]                 = src_pixel[1 - offset];
            yyvec_pixel[x]                = src_pixel[2 + offset];
            vvec_pixel[x]                 = src_pixel[3 - offset];
        }
        elem_coord.set(1, y + 1);
    }

    yuyv_to_rgb_calculation(yvec, vvec, yyvec, uvec, dst);
}

template <typename T>
inline void colorconvert_iyuv_to_rgb(const TensorShape &shape, const std::vector<SimpleTensor<T>> &tensor_planes, SimpleTensor<T> &dst)
{
    SimpleTensor<T> yvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> uvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> yyvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> vvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);

    Coordinates elem_coord{ 0, 0 };
    const int   yvec_width  = yvec.shape().x();
    const int   yvec_height = yvec.shape().y();

    for(int y = 0; y < yvec_height; ++y)
    {
        for(int x = 0; x < yvec_width; ++x)
        {
            const Coordinates src_coord{ x, y };
            const auto       *src_pixel   = reinterpret_cast<const T *>(tensor_planes[0](src_coord));
            auto             *yvec_pixel  = reinterpret_cast<T *>(yvec(elem_coord));
            auto             *yyvec_pixel = reinterpret_cast<T *>(yyvec(elem_coord));
            yvec_pixel[x]                 = src_pixel[x];
            yyvec_pixel[x]                = src_pixel[x + 1];
        }
        elem_coord.set(1, y + 1);
    }

    const int uvec_width  = uvec.shape().x();
    const int uvec_height = uvec.shape().y();

    Coordinates top_elem_coord{ 0, 0 };
    Coordinates bottom_elem_coord{ 0, 1 };
    for(int y = 0; y < uvec_height; y += 2)
    {
        for(int x = 0; x < uvec_width; ++x)
        {
            const Coordinates src_coord{ x, y / 2 };
            const auto       *src_pixel      = reinterpret_cast<const T *>(tensor_planes[1](src_coord));
            auto             *uvec_pixel_top = reinterpret_cast<T *>(uvec(top_elem_coord));
            auto             *vvec_pixel_top = reinterpret_cast<T *>(vvec(top_elem_coord));

            auto *uvec_pixel_bottom = reinterpret_cast<T *>(uvec(bottom_elem_coord));
            auto *vvec_pixel_bottom = reinterpret_cast<T *>(vvec(bottom_elem_coord));
            uvec_pixel_top[x]       = src_pixel[0];
            vvec_pixel_top[x]       = src_pixel[0];
            uvec_pixel_bottom[x]    = src_pixel[0];
            vvec_pixel_bottom[x]    = src_pixel[0];
        }
        top_elem_coord.set(1, y + 2);
        bottom_elem_coord.set(1, top_elem_coord.y() + 1);
    }

    yuyv_to_rgb_calculation(yvec, vvec, yyvec, uvec, dst);
}

template <typename T>
inline void colorconvert_nv12_to_rgb(const TensorShape &shape, const Format format, const std::vector<SimpleTensor<T>> &tensor_planes, SimpleTensor<T> &dst)
{
    SimpleTensor<T> yvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> uvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> yyvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> vvec(TensorShape{ tensor_planes[0].shape().x(), tensor_planes[0].shape().y() }, Format::U8);

    const int offset = (Format::NV12 == format) ? 0 : 1;

    Coordinates elem_coord{ 0, 0 };
    const int   yvec_width  = yvec.shape().x();
    const int   yvec_height = yvec.shape().y();

    for(int y = 0; y < yvec_height; ++y)
    {
        for(int x = 0; x < yvec_width; ++x)
        {
            const Coordinates src_coord{ x, y };
            const auto       *src_pixel   = reinterpret_cast<const T *>(tensor_planes[0](src_coord));
            auto             *yvec_pixel  = reinterpret_cast<T *>(yvec(elem_coord));
            auto             *yyvec_pixel = reinterpret_cast<T *>(yyvec(elem_coord));
            yvec_pixel[x]                 = src_pixel[x];
            yyvec_pixel[x]                = src_pixel[x + 1];
        }
        elem_coord.set(1, y + 1);
    }

    const int uvec_width  = uvec.shape().x();
    const int uvec_height = uvec.shape().y();

    Coordinates top_elem_coord{ 0, 0 };
    Coordinates bottom_elem_coord{ 0, 1 };
    for(int y = 0; y < uvec_height; y += 2)
    {
        for(int x = 0; x < uvec_width; ++x)
        {
            const Coordinates src_coord{ x, y / 2 };
            const auto       *src_pixel      = reinterpret_cast<const T *>(tensor_planes[1](src_coord));
            auto             *uvec_pixel_top = reinterpret_cast<T *>(uvec(top_elem_coord));
            auto             *vvec_pixel_top = reinterpret_cast<T *>(vvec(top_elem_coord));

            auto *uvec_pixel_bottom = reinterpret_cast<T *>(uvec(bottom_elem_coord));
            auto *vvec_pixel_bottom = reinterpret_cast<T *>(vvec(bottom_elem_coord));
            uvec_pixel_top[x]       = src_pixel[0 + offset];
            vvec_pixel_top[x]       = src_pixel[1 - offset];
            uvec_pixel_bottom[x]    = src_pixel[0 + offset];
            vvec_pixel_bottom[x]    = src_pixel[1 - offset];
        }
        top_elem_coord.set(1, y + 2);
        bottom_elem_coord.set(1, top_elem_coord.y() + 1);
    }

    yuyv_to_rgb_calculation(yvec, vvec, yyvec, uvec, dst);
}

} // namespace detail
} // color_convert_helper
} // namespace test
} // namespace arm_compute
#endif /*__ARM_COMPUTE_TEST_VALIDATION_COLOR_CONVERT_H__ */
