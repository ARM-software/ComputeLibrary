/*
 * Copyright (c) 2017-2019 ARM Limited.
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

constexpr float rgb2yuv_bt709_kr = 0.2126f;
constexpr float rgb2yuv_bt709_kb = 0.0722f;
// K_g = 1 - K_r - K_b
constexpr float rgb2yuv_bt709_kg = 0.7152f;
// C_u = 1 / (2 * (1 - K_b))
constexpr float rgb2yuv_bt709_cu = 0.5389f;
// C_v = 1 / (2 * (1 - K_r))
constexpr float rgb2yuv_bt709_cv = 0.6350f;

constexpr float rgb2u8_red_coef   = 0.2126f;
constexpr float rgb2u8_green_coef = 0.7152f;
constexpr float rgb2u8_blue_coef  = 0.0722f;

template <typename T>
inline void store_rgb_from_src(const SimpleTensor<T> src, SimpleTensor<T> &rvec, SimpleTensor<T> &gvec, SimpleTensor<T> &bvec)
{
    int width  = src.shape().x();
    int height = src.shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const Coordinates src_coord{ x, y };
            const Coordinates vec_coord{ x, y };

            const auto *src_pixel  = reinterpret_cast<const T *>(src(src_coord));
            auto       *rvec_pixel = reinterpret_cast<T *>(rvec(vec_coord));
            auto       *gvec_pixel = reinterpret_cast<T *>(gvec(vec_coord));
            auto       *bvec_pixel = reinterpret_cast<T *>(bvec(vec_coord));

            rvec_pixel[0] = src_pixel[0]; // NOLINT
            gvec_pixel[0] = src_pixel[1];
            bvec_pixel[0] = src_pixel[2];
        }
    }
}

template <typename T>
inline void rgb_to_yuv_calculation(const SimpleTensor<T> rvec, const SimpleTensor<T> gvec, const SimpleTensor<T> bvec, SimpleTensor<T> &yvec, SimpleTensor<T> &uvec_top, SimpleTensor<T> &uvec_bottom,
                                   SimpleTensor<T> &vvec_top, SimpleTensor<T> &vvec_bottom)
{
    int width  = rvec.shape().x();
    int height = rvec.shape().y();

    int         uvec_coord_x = 0;
    int         uvec_coord_y = 0;
    Coordinates uvec_coord{ uvec_coord_x, uvec_coord_y };

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x += 2)
        {
            Coordinates coord{ x, y };
            auto       *yvec_pixel        = reinterpret_cast<T *>(yvec(coord));
            auto       *uvec_top_pixel    = reinterpret_cast<T *>(uvec_top(uvec_coord));
            auto       *uvec_bottom_pixel = reinterpret_cast<T *>(uvec_bottom(uvec_coord));
            auto       *vvec_top_pixel    = reinterpret_cast<T *>(vvec_top(uvec_coord));
            auto       *vvec_bottom_pixel = reinterpret_cast<T *>(vvec_bottom(uvec_coord));

            T     border_value(0);
            int   rvec_val = validation::tensor_elem_at(rvec, coord, BorderMode::CONSTANT, border_value);
            int   gvec_val = validation::tensor_elem_at(gvec, coord, BorderMode::CONSTANT, border_value);
            int   bvec_val = validation::tensor_elem_at(bvec, coord, BorderMode::CONSTANT, border_value);
            float result   = rvec_val * rgb2yuv_bt709_kr + gvec_val * rgb2yuv_bt709_kg + bvec_val * rgb2yuv_bt709_kb;

            yvec_pixel[0]     = result;
            uvec_top_pixel[0] = (bvec_val - result) * rgb2yuv_bt709_cu + 128.f;
            vvec_top_pixel[0] = (rvec_val - result) * rgb2yuv_bt709_cv + 128.f;

            coord.set(0, x + 1);
            rvec_val = validation::tensor_elem_at(rvec, coord, BorderMode::CONSTANT, border_value);
            gvec_val = validation::tensor_elem_at(gvec, coord, BorderMode::CONSTANT, border_value);
            bvec_val = validation::tensor_elem_at(bvec, coord, BorderMode::CONSTANT, border_value);
            result   = rvec_val * rgb2yuv_bt709_kr + gvec_val * rgb2yuv_bt709_kg + bvec_val * rgb2yuv_bt709_kb;

            yvec_pixel[1]        = result;
            uvec_bottom_pixel[0] = (bvec_val - result) * rgb2yuv_bt709_cu + 128.f;
            vvec_bottom_pixel[0] = (rvec_val - result) * rgb2yuv_bt709_cv + 128.f;

            uvec_coord.set(0, ++uvec_coord_x);
        }
    }
}
inline float compute_rgb_value(int y_value, int v_value, int u_value, unsigned char channel_idx)
{
    float result = 0.f;
    switch(channel_idx)
    {
        case 0:
        {
            const float red = (v_value - 128.f) * red_coef_bt709;
            result          = y_value + red;
            break;
        }
        case 1:
        {
            const float green = (u_value - 128.f) * green_coef_bt709 + (v_value - 128.f) * green_coef2_bt709;
            result            = y_value + green;
            break;
        }
        case 2:
        {
            const float blue = (u_value - 128.f) * blue_coef_bt709;
            result           = y_value + blue;
            break;
        }
        default:
        {
            //Assuming Alpha channel
            return 255;
        }
    }
    return std::min(std::max(0.f, result), 255.f);
}

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
            const Coordinates dst_coord{ x, y };
            auto             *dst_pixel = reinterpret_cast<T *>(dst(dst_coord));
            const T           border_value(0);
            const int         yvec_val  = validation::tensor_elem_at(yvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int         vvec_val  = validation::tensor_elem_at(vvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int         yyvec_val = validation::tensor_elem_at(yyvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            const int         uvec_val  = validation::tensor_elem_at(uvec, { x_coord, y }, BorderMode::CONSTANT, border_value);
            //Compute first RGB value using Y plane
            for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
            {
                const float channel_value = compute_rgb_value(yvec_val, vvec_val, uvec_val, channel_idx);
                dst_pixel[channel_idx]    = channel_value;
            }
            //Compute second RGB value using YY plane
            const Coordinates dst_coord2
            {
                x + 1, y
            };
            dst_pixel = reinterpret_cast<T *>(dst(dst_coord2));
            for(int channel_idx = 0; channel_idx < dst.num_channels(); ++channel_idx)
            {
                const float channel_value = compute_rgb_value(yyvec_val, vvec_val, uvec_val, channel_idx);
                dst_pixel[channel_idx]    = channel_value;
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
inline void colorconvert_rgb_to_u8(const SimpleTensor<T> src, SimpleTensor<T> &dst)
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

            const float result = rgb2u8_red_coef * src_pixel[0] + rgb2u8_green_coef * src_pixel[1] + rgb2u8_blue_coef * src_pixel[2];

            dst_pixel[0] = utility::clamp<float>(result, 0, 255);
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
    SimpleTensor<T> yvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> uvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> yyvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> vvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);

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
            const auto       *u_pixel        = reinterpret_cast<const T *>(tensor_planes[1](src_coord));
            const auto       *v_pixel        = reinterpret_cast<const T *>(tensor_planes[2](src_coord));
            auto             *uvec_pixel_top = reinterpret_cast<T *>(uvec(top_elem_coord));
            auto             *vvec_pixel_top = reinterpret_cast<T *>(vvec(top_elem_coord));

            auto *uvec_pixel_bottom = reinterpret_cast<T *>(uvec(bottom_elem_coord));
            auto *vvec_pixel_bottom = reinterpret_cast<T *>(vvec(bottom_elem_coord));
            uvec_pixel_top[x]       = u_pixel[0];
            vvec_pixel_top[x]       = v_pixel[0];
            uvec_pixel_bottom[x]    = u_pixel[0];
            vvec_pixel_bottom[x]    = v_pixel[0];
        }
        top_elem_coord.set(1, y + 2);
        bottom_elem_coord.set(1, top_elem_coord.y() + 1);
    }

    yuyv_to_rgb_calculation(yvec, vvec, yyvec, uvec, dst);
}

template <typename T>
inline void colorconvert_nv12_to_rgb(const TensorShape &shape, const Format format, const std::vector<SimpleTensor<T>> &tensor_planes, SimpleTensor<T> &dst)
{
    SimpleTensor<T> yvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> uvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> yyvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);
    SimpleTensor<T> vvec(TensorShape{ tensor_planes[0].shape().x() / 2, tensor_planes[0].shape().y() }, Format::U8);

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

template <typename T>
inline void colorconvert_rgb_to_nv12(const SimpleTensor<T> src, std::vector<SimpleTensor<T>> &dst)
{
    SimpleTensor<T> rvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> gvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> bvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> yvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);

    int vec_shape_x = src.shape().x() * src.shape().y();

    SimpleTensor<T> uvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> uvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);

    store_rgb_from_src(src, rvec, gvec, bvec);
    rgb_to_yuv_calculation(rvec, gvec, bvec, dst[0], uvec_top, uvec_bottom, vvec_top, vvec_bottom);

    SimpleTensor<T> utmp(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    SimpleTensor<T> vtmp(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);

    int utmp_width  = utmp.shape().x();
    int utmp_height = utmp.shape().y();

    int         uvec_coord_x = 0;
    int         uvec_coord_y = 0;
    Coordinates uvec_coord{ uvec_coord_x, uvec_coord_y };
    for(int y = 0; y < utmp_height; y++)
    {
        for(int x = 0; x < utmp_width; x++)
        {
            Coordinates coord{ x, y };
            auto       *utmp_pixel = reinterpret_cast<T *>(utmp(coord));
            auto       *vtmp_pixel = reinterpret_cast<T *>(vtmp(coord));

            T   border_value(0);
            int uvec_top_val    = validation::tensor_elem_at(uvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int uvec_bottom_val = validation::tensor_elem_at(uvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);
            int vvec_top_val    = validation::tensor_elem_at(vvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int vvec_bottom_val = validation::tensor_elem_at(vvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);

            utmp_pixel[0] = std::ceil(float(uvec_top_val + uvec_bottom_val) / 2);
            vtmp_pixel[0] = std::ceil(float(vvec_top_val + vvec_bottom_val) / 2);

            uvec_coord.set(0, ++uvec_coord_x);
        }
    }

    int second_plane_x = dst[1].shape().x();
    int second_plane_y = dst[1].shape().y();

    int utmp_coord_x = 0;
    int utmp_coord_y = 0;

    for(int y = 0; y < second_plane_y; y++)
    {
        for(int x = 0; x < second_plane_x; x++)
        {
            Coordinates coord{ x, y };
            Coordinates utmp_top_coord{ utmp_coord_x, utmp_coord_y };
            Coordinates utmp_bottom_coord{ utmp_coord_x, utmp_coord_y + 1 };

            auto *dst_pixel = reinterpret_cast<T *>(dst[1](coord));

            T   border_value(0);
            int utmp_top_val    = validation::tensor_elem_at(utmp, utmp_top_coord, BorderMode::CONSTANT, border_value);
            int utmp_bottom_val = validation::tensor_elem_at(utmp, utmp_bottom_coord, BorderMode::CONSTANT, border_value);

            int result   = (utmp_top_val + utmp_bottom_val) / 2;
            dst_pixel[0] = result;

            int vtmp_top_val    = validation::tensor_elem_at(vtmp, utmp_top_coord, BorderMode::CONSTANT, border_value);
            int vtmp_bottom_val = validation::tensor_elem_at(vtmp, utmp_bottom_coord, BorderMode::CONSTANT, border_value);

            result       = (vtmp_top_val + vtmp_bottom_val) / 2;
            dst_pixel[1] = result;

            utmp_coord_x++;

            if(utmp_coord_x >= utmp_width)
            {
                utmp_coord_x = 0;
                utmp_coord_y += 2;
            }
        }
    }
}

template <typename T>
inline void colorconvert_rgb_to_iyuv(const SimpleTensor<T> src, std::vector<SimpleTensor<T>> &dst)
{
    SimpleTensor<T> rvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> gvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> bvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> yvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);

    int vec_shape_x = src.shape().x() * src.shape().y();

    SimpleTensor<T> uvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> uvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);

    store_rgb_from_src(src, rvec, gvec, bvec);
    rgb_to_yuv_calculation(rvec, gvec, bvec, dst[0], uvec_top, uvec_bottom, vvec_top, vvec_bottom);

    SimpleTensor<T> utmp(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    SimpleTensor<T> vtmp(TensorShape{ src.shape().x() / 2, src.shape().y() }, Format::U8);
    int             utmp_width  = utmp.shape().x();
    int             utmp_height = utmp.shape().y();

    int         uvec_coord_x = 0;
    int         uvec_coord_y = 0;
    Coordinates uvec_coord{ uvec_coord_x, uvec_coord_y };
    for(int y = 0; y < utmp_height; y++)
    {
        for(int x = 0; x < utmp_width; x++)
        {
            Coordinates coord{ x, y };
            auto       *utmp_pixel = reinterpret_cast<T *>(utmp(coord));
            auto       *vtmp_pixel = reinterpret_cast<T *>(vtmp(coord));

            T   border_value(0);
            int uvec_top_val    = validation::tensor_elem_at(uvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int uvec_bottom_val = validation::tensor_elem_at(uvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);
            int vvec_top_val    = validation::tensor_elem_at(vvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int vvec_bottom_val = validation::tensor_elem_at(vvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);

            utmp_pixel[0] = std::ceil(float(uvec_top_val + uvec_bottom_val) / 2);
            vtmp_pixel[0] = std::ceil(float(vvec_top_val + vvec_bottom_val) / 2);

            uvec_coord.set(0, ++uvec_coord_x);
        }
    }

    int second_plane_x = dst[1].shape().x();
    int second_plane_y = dst[1].shape().y();

    int utmp_coord_x = 0;
    int utmp_coord_y = 0;

    for(int y = 0; y < second_plane_y; y++)
    {
        for(int x = 0; x < second_plane_x; x++)
        {
            Coordinates coord{ x, y };
            Coordinates utmp_top_coord{ utmp_coord_x, utmp_coord_y };
            Coordinates utmp_bottom_coord{ utmp_coord_x, utmp_coord_y + 1 };

            auto *u_pixel = reinterpret_cast<T *>(dst[1](coord));
            auto *v_pixel = reinterpret_cast<T *>(dst[2](coord));

            T   border_value(0);
            int utmp_top_val    = validation::tensor_elem_at(utmp, utmp_top_coord, BorderMode::CONSTANT, border_value);
            int utmp_bottom_val = validation::tensor_elem_at(utmp, utmp_bottom_coord, BorderMode::CONSTANT, border_value);

            int result = (utmp_top_val + utmp_bottom_val) / 2;
            u_pixel[0] = result;

            int vtmp_top_val    = validation::tensor_elem_at(vtmp, utmp_top_coord, BorderMode::CONSTANT, border_value);
            int vtmp_bottom_val = validation::tensor_elem_at(vtmp, utmp_bottom_coord, BorderMode::CONSTANT, border_value);

            result     = (vtmp_top_val + vtmp_bottom_val) / 2;
            v_pixel[0] = result;

            utmp_coord_x++;

            if(utmp_coord_x >= utmp_width)
            {
                utmp_coord_x = 0;
                utmp_coord_y += 2;
            }
        }
    }
}

template <typename T>
inline void colorconvert_rgb_to_yuv4(const SimpleTensor<T> src, std::vector<SimpleTensor<T>> &dst)
{
    SimpleTensor<T> rvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> gvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> bvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);
    SimpleTensor<T> yvec(TensorShape{ dst[0].shape().x(), dst[0].shape().y() }, Format::U8);

    int vec_shape_x = src.shape().x() * src.shape().y();

    SimpleTensor<T> uvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> uvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_top(TensorShape{ vec_shape_x, 1U }, Format::U8);
    SimpleTensor<T> vvec_bottom(TensorShape{ vec_shape_x, 1U }, Format::U8);

    int width  = src.shape().x();
    int height = src.shape().y();

    store_rgb_from_src(src, rvec, gvec, bvec);

    rgb_to_yuv_calculation(rvec, gvec, bvec, dst[0], uvec_top, uvec_bottom, vvec_top, vvec_bottom);

    int         uvec_coord_x = 0;
    int         uvec_coord_y = 0;
    Coordinates uvec_coord{ uvec_coord_x, uvec_coord_y };
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x += 2)
        {
            Coordinates coord{ x, y };
            auto       *plane_1_pixel = reinterpret_cast<T *>(dst[1](coord));
            auto       *plane_2_pixel = reinterpret_cast<T *>(dst[2](coord));

            T   border_value(0);
            int uvec_top_val    = validation::tensor_elem_at(uvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int uvec_bottom_val = validation::tensor_elem_at(uvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);

            plane_1_pixel[0] = uvec_top_val;
            plane_1_pixel[1] = uvec_bottom_val;

            int vvec_top_val    = validation::tensor_elem_at(vvec_top, uvec_coord, BorderMode::CONSTANT, border_value);
            int vvec_bottom_val = validation::tensor_elem_at(vvec_bottom, uvec_coord, BorderMode::CONSTANT, border_value);

            plane_2_pixel[0] = vvec_top_val;
            plane_2_pixel[1] = vvec_bottom_val;

            uvec_coord.set(0, ++uvec_coord_x);
        }
    }
}

template <typename T>
inline void colorconvert_yuyv_to_nv12(const SimpleTensor<T> src, const Format format, std::vector<SimpleTensor<T>> &dst)
{
    SimpleTensor<T> uvvec_top(TensorShape{ dst[0].shape().x(), dst[0].shape().y() / 2 }, Format::U8);
    SimpleTensor<T> uvvec_bottom(TensorShape{ dst[0].shape().x(), dst[0].shape().y() / 2 }, Format::U8);

    const int offset = (Format::YUYV422 == format) ? 0 : 1;

    int width  = dst[0].shape().x();
    int height = dst[0].shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates dst_coord{ x, y };
            const Coordinates uv_coord{ x, y / 2 };

            const auto *src_pixel          = reinterpret_cast<const T *>(src(dst_coord));
            auto       *y_pixel            = reinterpret_cast<T *>(dst[0](dst_coord));
            auto       *uvvec_top_pixel    = reinterpret_cast<T *>(uvvec_top(uv_coord));
            auto       *uvvec_bottom_pixel = reinterpret_cast<T *>(uvvec_bottom(uv_coord));

            y_pixel[0] = src_pixel[0 + offset];

            if(y % 2 == 0)
            {
                uvvec_top_pixel[0] = src_pixel[1 - offset];
            }
            else
            {
                uvvec_bottom_pixel[0] = src_pixel[1 - offset];
            }
        }
    }

    width  = dst[1].shape().x();
    height = dst[1].shape().y();

    int uv_coord_x = 0;
    int uv_coord_y = 0;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates dst_coord{ x, y };
            const Coordinates uv_coord{ uv_coord_x, uv_coord_y };

            auto       *uv_pixel           = reinterpret_cast<T *>(dst[1](dst_coord));
            const auto *uvvec_top_pixel    = reinterpret_cast<T *>(uvvec_top(uv_coord));
            const auto *uvvec_bottom_pixel = reinterpret_cast<T *>(uvvec_bottom(uv_coord));

            uv_pixel[0] = (uvvec_top_pixel[0] + uvvec_bottom_pixel[0]) / 2;
            uv_pixel[1] = (uvvec_top_pixel[1] + uvvec_bottom_pixel[1]) / 2;
            uv_coord_x += 2;
        }
        uv_coord_x = 0;
        uv_coord_y++;
    }
}

template <typename T>
inline void colorconvert_yuyv_to_iyuv(const SimpleTensor<T> src, const Format format, std::vector<SimpleTensor<T>> &dst)
{
    SimpleTensor<T> uvvec_top(TensorShape{ dst[0].shape().x(), dst[0].shape().y() / 2 }, Format::U8);
    SimpleTensor<T> uvvec_bottom(TensorShape{ dst[0].shape().x(), dst[0].shape().y() / 2 }, Format::U8);

    const int offset = (Format::YUYV422 == format) ? 0 : 1;

    int width  = dst[0].shape().x();
    int height = dst[0].shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates dst_coord{ x, y };
            const Coordinates uv_coord{ x, y / 2 };

            const auto *src_pixel          = reinterpret_cast<const T *>(src(dst_coord));
            auto       *y_pixel            = reinterpret_cast<T *>(dst[0](dst_coord));
            auto       *uvvec_top_pixel    = reinterpret_cast<T *>(uvvec_top(uv_coord));
            auto       *uvvec_bottom_pixel = reinterpret_cast<T *>(uvvec_bottom(uv_coord));

            y_pixel[0] = src_pixel[0 + offset];

            if(y % 2 == 0)
            {
                uvvec_top_pixel[0] = src_pixel[1 - offset];
            }
            else
            {
                uvvec_bottom_pixel[0] = src_pixel[1 - offset];
            }
        }
    }

    width  = dst[1].shape().x();
    height = dst[1].shape().y();

    int uv_coord_x = 0;
    int uv_coord_y = 0;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates dst_coord{ x, y };
            const Coordinates uv_coord{ uv_coord_x, uv_coord_y };

            auto       *u_pixel            = reinterpret_cast<T *>(dst[1](dst_coord));
            auto       *v_pixel            = reinterpret_cast<T *>(dst[2](dst_coord));
            const auto *uvvec_top_pixel    = reinterpret_cast<T *>(uvvec_top(uv_coord));
            const auto *uvvec_bottom_pixel = reinterpret_cast<T *>(uvvec_bottom(uv_coord));

            u_pixel[0] = (uvvec_top_pixel[0] + uvvec_bottom_pixel[0]) / 2;
            v_pixel[0] = (uvvec_top_pixel[1] + uvvec_bottom_pixel[1]) / 2;
            uv_coord_x += 2;
        }
        uv_coord_x = 0;
        uv_coord_y++;
    }
}

template <typename T>
inline void nv_to_iyuv(const SimpleTensor<T> src, const Format src_format, SimpleTensor<T> &nv1, SimpleTensor<T> &nv2)
{
    int width  = src.shape().x();
    int height = src.shape().y();

    const int offset = (Format::NV12 == src_format) ? 1 : 0;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates src_coord{ x, y };
            const auto       *src_pixel = reinterpret_cast<const T *>(src(src_coord));
            auto             *u_pixel   = reinterpret_cast<T *>(nv1(src_coord));
            auto             *v_pixel   = reinterpret_cast<T *>(nv2(src_coord));

            u_pixel[0] = src_pixel[1 - offset];
            v_pixel[0] = src_pixel[0 + offset];
        }
    }
}

template <typename T>
inline void nv_to_yuv4(const SimpleTensor<T> src, const Format src_format, SimpleTensor<T> &nv1, SimpleTensor<T> &nv2)
{
    int width  = src.shape().x();
    int height = src.shape().y();

    const int offset = (Format::NV12 == src_format) ? 1 : 0;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; x++)
        {
            const Coordinates src_coord{ x, y };
            Coordinates       dst_coord{ x * 2, y * 2 };
            const auto       *src_pixel = reinterpret_cast<const T *>(src(src_coord));
            auto             *u_pixel   = reinterpret_cast<T *>(nv1(dst_coord));
            auto             *v_pixel   = reinterpret_cast<T *>(nv2(dst_coord));

            u_pixel[0] = src_pixel[1 - offset];
            u_pixel[1] = src_pixel[1 - offset];

            v_pixel[0] = src_pixel[0 + offset];
            v_pixel[1] = src_pixel[0 + offset];

            dst_coord.set(1, y * 2 + 1);
            u_pixel    = reinterpret_cast<T *>(nv1(dst_coord));
            v_pixel    = reinterpret_cast<T *>(nv2(dst_coord));
            u_pixel[0] = src_pixel[1 - offset];
            u_pixel[1] = src_pixel[1 - offset];

            v_pixel[0] = src_pixel[0 + offset];
            v_pixel[1] = src_pixel[0 + offset];
        }
    }
}

template <typename T>
inline void colorconvert_nv_to_iyuv(const std::vector<SimpleTensor<T>> src, const Format src_format, std::vector<SimpleTensor<T>> &dst)
{
    int width  = dst[0].shape().x();
    int height = dst[0].shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const Coordinates dst_coord{ x, y };

            const auto *src_pixel = reinterpret_cast<const T *>(src[0](dst_coord));
            auto       *y_pixel   = reinterpret_cast<T *>(dst[0](dst_coord));

            y_pixel[0] = src_pixel[0];
        }
    }

    nv_to_iyuv(src[1], src_format, dst[1], dst[2]);
}

template <typename T>
inline void colorconvert_nv_to_yuv4(const std::vector<SimpleTensor<T>> src, const Format src_format, std::vector<SimpleTensor<T>> &dst)
{
    int width  = dst[0].shape().x();
    int height = dst[0].shape().y();

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const Coordinates dst_coord{ x, y };

            const auto *src_pixel = reinterpret_cast<const T *>(src[0](dst_coord));
            auto       *y_pixel   = reinterpret_cast<T *>(dst[0](dst_coord));

            y_pixel[0] = src_pixel[0];
        }
    }

    nv_to_yuv4(src[1], src_format, dst[1], dst[2]);
}

} // namespace detail
} // color_convert_helper
} // namespace test
} // namespace arm_compute
#endif /*__ARM_COMPUTE_TEST_VALIDATION_COLOR_CONVERT_H__ */
