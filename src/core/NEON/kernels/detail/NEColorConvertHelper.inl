/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/Utils.h"
#include "src/core/NEON/NEMath.h"

#include <arm_neon.h>

namespace
{
#ifndef DOXYGEN_SKIP_THIS
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

inline float32x4_t rgb_to_greyscale_calculation(const float32x4_t &rcolor, const float32x4_t &gcolor, const float32x4_t &bcolor,
                                                const float rcoef, const float gcoef, const float bcoef)
{
    float32x4_t greyscale = vmulq_n_f32(rcolor, rcoef);
    greyscale             = vmlaq_n_f32(greyscale, gcolor, gcoef);
    greyscale             = vmlaq_n_f32(greyscale, bcolor, bcoef);
    return greyscale;
}

inline void rgb_to_u8_conversion(const uint8x16x3_t &in, uint8x16_t &out)
{
    float32x4x4_t out_float32;

    //Conversion from 3(RGB) 4 uint8s to 3(RGB) 4 floats
    const float32x4x4_t r_float32 = arm_compute::convert_uint8x16_to_float32x4x4(in.val[0]);
    const float32x4x4_t g_float32 = arm_compute::convert_uint8x16_to_float32x4x4(in.val[1]);
    const float32x4x4_t b_float32 = arm_compute::convert_uint8x16_to_float32x4x4(in.val[2]);

    //New grayscale image = ( (RED_COEFF * R) + (GREEN_COEFF * G) + (BLUE_COEFF * B) )
    //Computation of 1(Greyscale) 4 uint8 using 3(RGB) 4 uint8s float
    out_float32.val[0] = rgb_to_greyscale_calculation(r_float32.val[0], g_float32.val[0], b_float32.val[0],
                                                      rgb2u8_red_coef, rgb2u8_green_coef, rgb2u8_blue_coef);

    out_float32.val[1] = rgb_to_greyscale_calculation(r_float32.val[1], g_float32.val[1], b_float32.val[1],
                                                      rgb2u8_red_coef, rgb2u8_green_coef, rgb2u8_blue_coef);

    out_float32.val[2] = rgb_to_greyscale_calculation(r_float32.val[2], g_float32.val[2], b_float32.val[2],
                                                      rgb2u8_red_coef, rgb2u8_green_coef, rgb2u8_blue_coef);

    out_float32.val[3] = rgb_to_greyscale_calculation(r_float32.val[3], g_float32.val[3], b_float32.val[3],
                                                      rgb2u8_red_coef, rgb2u8_green_coef, rgb2u8_blue_coef);

    //Conversion from 1(Greyscale) 4 floats to 1(Greyscale) 4 uint8s
    arm_compute::convert_float32x4x4_to_uint8x16(out_float32, out);
}

inline void rgb_to_yuv_calculation(const float32x4_t &rvec, const float32x4_t &gvec, const float32x4_t &bvec,
                                   float32x4_t &yvec, float32x4_t &uvec, float32x4_t &vvec)
{
    /*
    Y'= 0.2126*R' + 0.7152*G' + 0.0722*B'
    U'=-0.1146*R' - 0.3854*G' + 0.5000*B'
    V'= 0.5000*R' - 0.4542*G' - 0.0458*B'
    */
    const auto c128 = vdupq_n_f32(128.f);

    // Y = R * K_r + G * (1 - K_r - K_b) * B * K_b
    yvec = vmulq_n_f32(rvec, rgb2yuv_bt709_kr);
    yvec = vmlaq_n_f32(yvec, gvec, rgb2yuv_bt709_kg);
    yvec = vmlaq_n_f32(yvec, bvec, rgb2yuv_bt709_kb);

    // U = (B - Y) / (2 * (1 - K_b))
    uvec = vsubq_f32(bvec, yvec);
    uvec = vmlaq_n_f32(c128, uvec, rgb2yuv_bt709_cu);

    // V = (R - Y) / (2 * (1 - K_r))
    vvec = vsubq_f32(rvec, yvec);
    vvec = vmlaq_n_f32(c128, vvec, rgb2yuv_bt709_cv);
}

inline void yuyv_to_rgb_calculation(const float32x4_t &yvec_val, float32x4_t uvec_val, const float32x4_t &yyvec_val,
                                    float32x4_t vvec_val, unsigned char *output_ptr, const bool alpha)
{
    float32x4x3_t rgb1, rgb2;

    // Compute: cb - 128 and cr - 128;
    const auto c128 = vdupq_n_f32(128.f);
    uvec_val        = vsubq_f32(uvec_val, c128);
    vvec_val        = vsubq_f32(vvec_val, c128);

    // Compute:
    // r = 0.0000f*f_u + 1.5748f*f_v;
    // g = 0.1873f*f_u - 0.4681f*f_v;
    // b = 1.8556f*f_u + 0.0000f*f_v;
    const auto red   = vmulq_n_f32(vvec_val, red_coef_bt709);
    const auto blue  = vmulq_n_f32(uvec_val, blue_coef_bt709);
    const auto green = vaddq_f32(vmulq_n_f32(uvec_val, green_coef_bt709),
                                 vmulq_n_f32(vvec_val, green_coef2_bt709));

    // Compute the final r,g,b values using y1 for the first texel and y2 for the second one.
    // the result is stored in two float32x4x3_t which then are converted to one uint8x8x3_t
    // and written back to memory using vst3 instruction

    rgb1.val[0] = vaddq_f32(yvec_val, red);
    rgb1.val[1] = vaddq_f32(yvec_val, green);
    rgb1.val[2] = vaddq_f32(yvec_val, blue);

    rgb2.val[0] = vaddq_f32(yyvec_val, red);
    rgb2.val[1] = vaddq_f32(yyvec_val, green);
    rgb2.val[2] = vaddq_f32(yyvec_val, blue);

    uint8x8x3_t u8_rgb;
    arm_compute::convert_float32x4x3_to_uint8x8x3(rgb1, rgb2, u8_rgb);

    if(!alpha)
    {
        vst3_lane_u8(&output_ptr[0], u8_rgb, 0);
        vst3_lane_u8(&output_ptr[3], u8_rgb, 4);
        vst3_lane_u8(&output_ptr[6], u8_rgb, 1);
        vst3_lane_u8(&output_ptr[9], u8_rgb, 5);
        vst3_lane_u8(&output_ptr[12], u8_rgb, 2);
        vst3_lane_u8(&output_ptr[15], u8_rgb, 6);
        vst3_lane_u8(&output_ptr[18], u8_rgb, 3);
        vst3_lane_u8(&output_ptr[21], u8_rgb, 7);
    }
    else
    {
        uint8x8x4_t u8_rgba;
        u8_rgba.val[0] = u8_rgb.val[0];
        u8_rgba.val[1] = u8_rgb.val[1];
        u8_rgba.val[2] = u8_rgb.val[2];
        u8_rgba.val[3] = vdup_n_u8(255);
        vst4_lane_u8(&output_ptr[0], u8_rgba, 0);
        vst4_lane_u8(&output_ptr[4], u8_rgba, 4);
        vst4_lane_u8(&output_ptr[8], u8_rgba, 1);
        vst4_lane_u8(&output_ptr[12], u8_rgba, 5);
        vst4_lane_u8(&output_ptr[16], u8_rgba, 2);
        vst4_lane_u8(&output_ptr[20], u8_rgba, 6);
        vst4_lane_u8(&output_ptr[24], u8_rgba, 3);
        vst4_lane_u8(&output_ptr[28], u8_rgba, 7);
    }
}

inline uint8x16x3_t load_rgb(const unsigned char *const ptr, const bool alpha)
{
    uint8x16x3_t rgb;

    if(alpha)
    {
        const auto tmp = vld4q_u8(ptr);
        rgb.val[0]     = tmp.val[0];
        rgb.val[1]     = tmp.val[1];
        rgb.val[2]     = tmp.val[2];
    }
    else
    {
        rgb = vld3q_u8(ptr);
    }

    return rgb;
}

inline void rgb_to_yuv_conversion(uint8x16x3_t &vec_top, uint8x16x3_t &vec_bottom)
{
    // Convert the uint8x16_t to float32x4x4_t
    const float32x4x4_t frvec_top = arm_compute::convert_uint8x16_to_float32x4x4(vec_top.val[0]);
    const float32x4x4_t fgvec_top = arm_compute::convert_uint8x16_to_float32x4x4(vec_top.val[1]);
    const float32x4x4_t fbvec_top = arm_compute::convert_uint8x16_to_float32x4x4(vec_top.val[2]);

    const float32x4x4_t frvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(vec_bottom.val[0]);
    const float32x4x4_t fgvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(vec_bottom.val[1]);
    const float32x4x4_t fbvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(vec_bottom.val[2]);

    float32x4x4_t fyvec_top, fuvec_top, fvvec_top;
    float32x4x4_t fyvec_bottom, fuvec_bottom, fvvec_bottom;

    for(auto i = 0; i < 4; ++i)
    {
        rgb_to_yuv_calculation(frvec_top.val[i], fgvec_top.val[i], fbvec_top.val[i],
                               fyvec_top.val[i], fuvec_top.val[i], fvvec_top.val[i]);
        rgb_to_yuv_calculation(frvec_bottom.val[i], fgvec_bottom.val[i], fbvec_bottom.val[i],
                               fyvec_bottom.val[i], fuvec_bottom.val[i], fvvec_bottom.val[i]);
    }

    arm_compute::convert_float32x4x4_to_uint8x16(fyvec_top, vec_top.val[0]);
    arm_compute::convert_float32x4x4_to_uint8x16(fuvec_top, vec_top.val[1]);
    arm_compute::convert_float32x4x4_to_uint8x16(fvvec_top, vec_top.val[2]);
    arm_compute::convert_float32x4x4_to_uint8x16(fyvec_bottom, vec_bottom.val[0]);
    arm_compute::convert_float32x4x4_to_uint8x16(fuvec_bottom, vec_bottom.val[1]);
    arm_compute::convert_float32x4x4_to_uint8x16(fvvec_bottom, vec_bottom.val[2]);
}

inline void store_rgb_to_nv12(const uint8x16_t &rvec_top, const uint8x16_t &gvec_top, const uint8x16_t &bvec_top,
                              const uint8x16_t &rvec_bottom, const uint8x16_t &gvec_bottom, const uint8x16_t &bvec_bottom,
                              unsigned char *const __restrict out_y_top, unsigned char *const __restrict out_y_bottom,
                              unsigned char *const __restrict out_uv)
{
    uint8x16x3_t vec_top, vec_bottom;
    vec_top.val[0]    = rvec_top;
    vec_top.val[1]    = gvec_top;
    vec_top.val[2]    = bvec_top;
    vec_bottom.val[0] = rvec_bottom;
    vec_bottom.val[1] = gvec_bottom;
    vec_bottom.val[2] = bvec_bottom;

    rgb_to_yuv_conversion(vec_top, vec_bottom);

    vst1q_u8(out_y_top, vec_top.val[0]);
    vst1q_u8(out_y_bottom, vec_bottom.val[0]);

    const auto uvec = vuzpq_u8(vec_top.val[1], vec_bottom.val[1]);
    const auto vvec = vuzpq_u8(vec_top.val[2], vec_bottom.val[2]);
    const auto utmp = vrhaddq_u8(uvec.val[0], uvec.val[1]);
    const auto vtmp = vrhaddq_u8(vvec.val[0], vvec.val[1]);

    uint8x8x2_t uvvec;
    uvvec.val[0] = vhadd_u8(vget_low_u8(utmp), vget_high_u8(utmp));
    uvvec.val[1] = vhadd_u8(vget_low_u8(vtmp), vget_high_u8(vtmp));

    vst2_u8(out_uv, uvvec);
}

inline void store_rgb_to_iyuv(const uint8x16_t &rvec_top, const uint8x16_t &gvec_top, const uint8x16_t &bvec_top,
                              const uint8x16_t &rvec_bottom, const uint8x16_t &gvec_bottom, const uint8x16_t &bvec_bottom,
                              unsigned char *const __restrict out_y_top, unsigned char *const __restrict out_y_bottom,
                              unsigned char *const __restrict out_u,
                              unsigned char *const __restrict out_v)
{
    uint8x16x3_t vec_top, vec_bottom;
    vec_top.val[0]    = rvec_top;
    vec_top.val[1]    = gvec_top;
    vec_top.val[2]    = bvec_top;
    vec_bottom.val[0] = rvec_bottom;
    vec_bottom.val[1] = gvec_bottom;
    vec_bottom.val[2] = bvec_bottom;

    rgb_to_yuv_conversion(vec_top, vec_bottom);

    vst1q_u8(out_y_top, vec_top.val[0]);
    vst1q_u8(out_y_bottom, vec_bottom.val[0]);

    const auto uvvec_top    = vuzpq_u8(vec_top.val[1], vec_top.val[2]);
    const auto uvvec_bottom = vuzpq_u8(vec_bottom.val[1], vec_bottom.val[2]);
    const auto uvvec        = vhaddq_u8(vrhaddq_u8(uvvec_top.val[0], uvvec_top.val[1]),
                                        vrhaddq_u8(uvvec_bottom.val[0], uvvec_bottom.val[1]));

    vst1_u8(out_u, vget_low_u8(uvvec));
    vst1_u8(out_v, vget_high_u8(uvvec));
}

inline void store_rgb_to_yuv4(const uint8x16_t &rvec, const uint8x16_t &gvec, const uint8x16_t &bvec,
                              unsigned char *const __restrict out_y,
                              unsigned char *const __restrict out_u,
                              unsigned char *const __restrict out_v)
{
    // Convert the uint8x16_t to float32x4x4_t
    const float32x4x4_t frvec = arm_compute::convert_uint8x16_to_float32x4x4(rvec);
    const float32x4x4_t fgvec = arm_compute::convert_uint8x16_to_float32x4x4(gvec);
    const float32x4x4_t fbvec = arm_compute::convert_uint8x16_to_float32x4x4(bvec);

    float32x4x4_t fyvec, fuvec, fvvec;
    for(auto i = 0; i < 4; ++i)
    {
        rgb_to_yuv_calculation(frvec.val[i], fgvec.val[i], fbvec.val[i],
                               fyvec.val[i], fuvec.val[i], fvvec.val[i]);
    }

    uint8x16_t yvec, uvec, vvec;
    arm_compute::convert_float32x4x4_to_uint8x16(fyvec, yvec);
    arm_compute::convert_float32x4x4_to_uint8x16(fuvec, uvec);
    arm_compute::convert_float32x4x4_to_uint8x16(fvvec, vvec);

    vst1q_u8(out_y, yvec);
    vst1q_u8(out_u, uvec);
    vst1q_u8(out_v, vvec);
}
#endif /* DOXYGEN_SKIP_THIS */
}

namespace arm_compute
{
/** Convert RGB to RGBX.
 *
 * @param[in]  input  Input RGB data buffer.
 * @param[out] output Output RGBX buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
void colorconvert_rgb_to_rgbx(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    Iterator in(input_ptr, win);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto   ta1 = vld3q_u8(in.ptr());
        uint8x16x4_t ta2;
        ta2.val[0] = ta1.val[0];
        ta2.val[1] = ta1.val[1];
        ta2.val[2] = ta1.val[2];
        ta2.val[3] = vdupq_n_u8(255);
        vst4q_u8(out.ptr(), ta2);
    },
    in, out);
}

/** Convert RGB to U8.
 *
 * @param[in]  input  Input RGB data buffer.
 * @param[out] output Output U8 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
void colorconvert_rgb_to_u8(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    Iterator in(input_ptr, win);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta1 = vld3q_u8(in.ptr());
        uint8x16_t ta2;
        rgb_to_u8_conversion(ta1, ta2);
        vst1q_u8(out.ptr(), ta2);
    },
    in, out);
}

/** Convert RGBX to RGB.
 *
 * @param[in]  input  Input RGBX data buffer.
 * @param[out] output Output RGB buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
void colorconvert_rgbx_to_rgb(const void *input, void *output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    Iterator in(input_ptr, win);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto   ta1 = vld4q_u8(in.ptr());
        uint8x16x3_t ta2;
        ta2.val[0] = ta1.val[0];
        ta2.val[1] = ta1.val[1];
        ta2.val[2] = ta1.val[2];
        vst3q_u8(out.ptr(), ta2);
    },
    in, out);
}

/** Convert YUYV to RGB.
 *
 * @param[in]  input  Input YUYV data buffer.
 * @param[out] output Output RGB buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool yuyv, bool alpha>
void colorconvert_yuyv_to_rgb(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    constexpr auto element_size = alpha ? 32 : 24;
    constexpr auto shift        = yuyv ? 0 : 1;

    Iterator in(input_ptr, win);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta = vld4q_u8(in.ptr());
        //ta.val[0] = Y0 Y2 Y4 Y6 ...
        //ta.val[1] = U0 U2 U4 U6 ...
        //ta.val[2] = Y1 Y3 Y5 Y7 ...
        //ta.val[3] = V0 V2 V4 V7 ...

        // Convert the uint8x16x4_t to float32x4x4_t
        const float32x4x4_t yvec  = arm_compute::convert_uint8x16_to_float32x4x4(ta.val[0 + shift]);
        const float32x4x4_t uvec  = arm_compute::convert_uint8x16_to_float32x4x4(ta.val[1 - shift]);
        const float32x4x4_t yyvec = arm_compute::convert_uint8x16_to_float32x4x4(ta.val[2 + shift]);
        const float32x4x4_t vvec  = arm_compute::convert_uint8x16_to_float32x4x4(ta.val[3 - shift]);

        yuyv_to_rgb_calculation(yvec.val[0], uvec.val[0], yyvec.val[0], vvec.val[0], out.ptr() + 0 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec.val[1], uvec.val[1], yyvec.val[1], vvec.val[1], out.ptr() + 1 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec.val[2], uvec.val[2], yyvec.val[2], vvec.val[2], out.ptr() + 2 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec.val[3], uvec.val[3], yyvec.val[3], vvec.val[3], out.ptr() + 3 * element_size, alpha);
    },
    in, out);
}

/** Convert NV12 to RGB.
 *
 * @param[in]  input  Input NV12 data buffer.
 * @param[out] output Output RGB buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool uv, bool alpha>
void colorconvert_nv12_to_rgb(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    constexpr auto element_size = alpha ? 32 : 24;
    const auto     out_stride   = output_ptr->info()->strides_in_bytes().y();
    constexpr auto shift        = uv ? 0 : 1;

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_uv(input_ptr->plane(1), win_uv);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_y_top    = vld2q_u8(in_y.ptr());
        const auto ta_y_bottom = vld2q_u8(in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y());
        const auto ta_uv       = vld2q_u8(in_uv.ptr());
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_uv.val[0] = U0 U2 U4 U6 ...
        //ta_uv.val[1] = V0 V2 V4 V6 ...

        // Convert the uint8x16x4_t to float32x4x4_t
        float32x4x4_t yvec_top     = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_top.val[0]);
        float32x4x4_t yyvec_top    = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_top.val[1]);
        float32x4x4_t yvec_bottom  = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_bottom.val[0]);
        float32x4x4_t yyvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_bottom.val[1]);
        float32x4x4_t uvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_uv.val[0 + shift]);
        float32x4x4_t vvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_uv.val[1 - shift]);

        yuyv_to_rgb_calculation(yvec_top.val[0], uvec.val[0], yyvec_top.val[0], vvec.val[0], out.ptr() + 0 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[1], uvec.val[1], yyvec_top.val[1], vvec.val[1], out.ptr() + 1 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[2], uvec.val[2], yyvec_top.val[2], vvec.val[2], out.ptr() + 2 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[3], uvec.val[3], yyvec_top.val[3], vvec.val[3], out.ptr() + 3 * element_size, alpha);

        yuyv_to_rgb_calculation(yvec_bottom.val[0], uvec.val[0], yyvec_bottom.val[0], vvec.val[0], out.ptr() + out_stride + 0 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[1], uvec.val[1], yyvec_bottom.val[1], vvec.val[1], out.ptr() + out_stride + 1 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[2], uvec.val[2], yyvec_bottom.val[2], vvec.val[2], out.ptr() + out_stride + 2 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[3], uvec.val[3], yyvec_bottom.val[3], vvec.val[3], out.ptr() + out_stride + 3 * element_size, alpha);
    },
    in_y, in_uv, out);
}

/** Convert IYUV to RGB.
 *
 * @param[in]  input  Input IYUV data buffer.
 * @param[out] output Output RGB buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool alpha>
void colorconvert_iyuv_to_rgb(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IImage *__restrict>(output);

    constexpr auto element_size = alpha ? 32 : 24;
    const auto     out_stride   = output_ptr->info()->strides_in_bytes().y();

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_u(input_ptr->plane(1), win_uv);
    Iterator in_v(input_ptr->plane(2), win_uv);
    Iterator out(output_ptr, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto *y_top_ptr    = in_y.ptr();
        const auto *y_bottom_ptr = in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y();
        const auto *u_ptr        = in_u.ptr();
        const auto *v_ptr        = in_v.ptr();

        // Work-around issue in gcc 9(>=) where vld2q might cause issues with register allocation
#if defined(__arch64__)
        const auto ta0_y_top    = vld1q_u8(y_top_ptr);
        const auto ta1_y_top    = vld1q_u8(y_top_ptr + 16);
        const auto ta0_y_bottom = vld1q_u8(y_bottom_ptr);
        const auto ta1_y_bottom = vld1q_u8(y_bottom_ptr + 16);
        const auto ta_u         = vld1q_u8(u_ptr);
        const auto ta_v         = vld1q_u8(v_ptr);

        // Convert the uint8x16x4_t to float32x4x4_t
        float32x4x4_t yvec_top     = arm_compute::convert_uint8x16_to_float32x4x4(vuzp1q_u8(ta0_y_top, ta1_y_top));
        float32x4x4_t yyvec_top    = arm_compute::convert_uint8x16_to_float32x4x4(vuzp2q_u8(ta0_y_top, ta1_y_top));
        float32x4x4_t yvec_bottom  = arm_compute::convert_uint8x16_to_float32x4x4(vuzp1q_u8(ta0_y_bottom, ta1_y_bottom));
        float32x4x4_t yyvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(vuzp2q_u8(ta0_y_bottom, ta1_y_bottom));
        float32x4x4_t uvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_u);
        float32x4x4_t vvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_v);
#else  /* defined(__arch64__) */
        const auto ta_y_top    = vld2q_u8(y_top_ptr);
        const auto ta_y_bottom = vld2q_u8(y_bottom_ptr);
        const auto ta_u        = vld1q_u8(u_ptr);
        const auto ta_v        = vld1q_u8(v_ptr);
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_u.val[0] = U0 U2 U4 U6 ...
        //ta_v.val[0] = V0 V2 V4 V6 ...

        // Convert the uint8x16x4_t to float32x4x4_t
        float32x4x4_t yvec_top     = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_top.val[0]);
        float32x4x4_t yyvec_top    = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_top.val[1]);
        float32x4x4_t yvec_bottom  = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_bottom.val[0]);
        float32x4x4_t yyvec_bottom = arm_compute::convert_uint8x16_to_float32x4x4(ta_y_bottom.val[1]);
        float32x4x4_t uvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_u);
        float32x4x4_t vvec         = arm_compute::convert_uint8x16_to_float32x4x4(ta_v);
#endif /* defined(__arch64__) */

        yuyv_to_rgb_calculation(yvec_top.val[0], uvec.val[0], yyvec_top.val[0], vvec.val[0], out.ptr() + 0 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[1], uvec.val[1], yyvec_top.val[1], vvec.val[1], out.ptr() + 1 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[2], uvec.val[2], yyvec_top.val[2], vvec.val[2], out.ptr() + 2 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_top.val[3], uvec.val[3], yyvec_top.val[3], vvec.val[3], out.ptr() + 3 * element_size, alpha);

        yuyv_to_rgb_calculation(yvec_bottom.val[0], uvec.val[0], yyvec_bottom.val[0], vvec.val[0], out.ptr() + out_stride + 0 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[1], uvec.val[1], yyvec_bottom.val[1], vvec.val[1], out.ptr() + out_stride + 1 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[2], uvec.val[2], yyvec_bottom.val[2], vvec.val[2], out.ptr() + out_stride + 2 * element_size, alpha);
        yuyv_to_rgb_calculation(yvec_bottom.val[3], uvec.val[3], yyvec_bottom.val[3], vvec.val[3], out.ptr() + out_stride + 3 * element_size, alpha);
    },
    in_y, in_u, in_v, out);
}

/** Convert YUYV to NV12.
 *
 * @param[in]  input  Input YUYV data buffer.
 * @param[out] output Output NV12 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool yuyv>
void colorconvert_yuyv_to_nv12(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    constexpr auto shift = yuyv ? 0 : 1;

    // NV12's UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in(input_ptr, win);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_uv(output_ptr->plane(1), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_top    = vld4q_u8(in.ptr());
        const auto ta_bottom = vld4q_u8(in.ptr() + input_ptr->info()->strides_in_bytes().y());
        //ta.val[0] = Y0 Y2 Y4 Y6 ...
        //ta.val[1] = U0 U2 U4 U6 ...
        //ta.val[2] = Y1 Y3 Y5 Y7 ...
        //ta.val[3] = V0 V2 V4 V7 ...

        uint8x16x2_t yvec;
        yvec.val[0] = ta_top.val[0 + shift];
        yvec.val[1] = ta_top.val[2 + shift];
        vst2q_u8(out_y.ptr(), yvec);

        uint8x16x2_t yyvec;
        yyvec.val[0] = ta_bottom.val[0 + shift];
        yyvec.val[1] = ta_bottom.val[2 + shift];
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), yyvec);

        uint8x16x2_t uvvec;
        uvvec.val[0] = vhaddq_u8(ta_top.val[1 - shift], ta_bottom.val[1 - shift]);
        uvvec.val[1] = vhaddq_u8(ta_top.val[3 - shift], ta_bottom.val[3 - shift]);
        vst2q_u8(out_uv.ptr(), uvvec);
    },
    in, out_y, out_uv);
}

/** Convert IYUV to NV12.
 *
 * @param[in]  input  Input IYUV data buffer.
 * @param[out] output Output NV12 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
void colorconvert_iyuv_to_nv12(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_u(input_ptr->plane(1), win_uv);
    Iterator in_v(input_ptr->plane(2), win_uv);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_uv(output_ptr->plane(1), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto   ta_y_top    = vld2q_u8(in_y.ptr());
        const auto   ta_y_bottom = vld2q_u8(in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y());
        uint8x16x2_t ta_uv;
        ta_uv.val[0] = vld1q_u8(in_u.ptr());
        ta_uv.val[1] = vld1q_u8(in_v.ptr());
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_uv.val[0] = U0 U2 U4 U6 ...
        //ta_uv.val[1] = V0 V2 V4 V6 ...

        vst2q_u8(out_y.ptr(), ta_y_top);
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), ta_y_bottom);
        vst2q_u8(out_uv.ptr(), ta_uv);
    },
    in_y, in_u, in_v, out_y, out_uv);
}

/** Convert NV12 to IYUV.
 *
 * @param[in]  input  Input NV12 data buffer.
 * @param[out] output Output IYUV buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool uv>
void colorconvert_nv12_to_iyuv(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    constexpr auto shift = uv ? 0 : 1;

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_uv(input_ptr->plane(1), win_uv);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win_uv);
    Iterator out_v(output_ptr->plane(2), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_y_top    = vld2q_u8(in_y.ptr());
        const auto ta_y_bottom = vld2q_u8(in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y());
        const auto ta_uv       = vld2q_u8(in_uv.ptr());
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_uv.val[0] = U0 U2 U4 U6 ...
        //ta_uv.val[1] = V0 V2 V4 V6 ...

        vst2q_u8(out_y.ptr(), ta_y_top);
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), ta_y_bottom);
        vst1q_u8(out_u.ptr(), ta_uv.val[0 + shift]);
        vst1q_u8(out_v.ptr(), ta_uv.val[1 - shift]);
    },
    in_y, in_uv, out_y, out_u, out_v);
}

/** Convert YUYV to IYUV.
 *
 * @param[in]  input  Input YUYV data buffer.
 * @param[out] output Output IYUV buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool yuyv>
void colorconvert_yuyv_to_iyuv(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    constexpr auto shift = yuyv ? 0 : 1;

    // Destination's UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in(input_ptr, win);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win_uv);
    Iterator out_v(output_ptr->plane(2), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_top    = vld4q_u8(in.ptr());
        const auto ta_bottom = vld4q_u8(in.ptr() + input_ptr->info()->strides_in_bytes().y());
        //ta.val[0] = Y0 Y2 Y4 Y6 ...
        //ta.val[1] = U0 U2 U4 U6 ...
        //ta.val[2] = Y1 Y3 Y5 Y7 ...
        //ta.val[3] = V0 V2 V4 V7 ...

        uint8x16x2_t yvec;
        yvec.val[0] = ta_top.val[0 + shift];
        yvec.val[1] = ta_top.val[2 + shift];
        vst2q_u8(out_y.ptr(), yvec);

        uint8x16x2_t yyvec;
        yyvec.val[0] = ta_bottom.val[0 + shift];
        yyvec.val[1] = ta_bottom.val[2 + shift];
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), yyvec);

        uint8x16_t uvec;
        uvec = vhaddq_u8(ta_top.val[1 - shift], ta_bottom.val[1 - shift]);
        vst1q_u8(out_u.ptr(), uvec);

        uint8x16_t vvec;
        vvec = vhaddq_u8(ta_top.val[3 - shift], ta_bottom.val[3 - shift]);
        vst1q_u8(out_v.ptr(), vvec);
    },
    in, out_y, out_u, out_v);
}

/** Convert NV12 to YUV4.
 *
 * @param[in]  input  Input NV12 data buffer.
 * @param[out] output Output YUV4 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool uv>
void colorconvert_nv12_to_yuv4(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    constexpr auto shift = uv ? 0 : 1;

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_uv(input_ptr->plane(1), win_uv);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win);
    Iterator out_v(output_ptr->plane(2), win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_y_top    = vld2q_u8(in_y.ptr());
        const auto ta_y_bottom = vld2q_u8(in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y());
        const auto ta_uv       = vld2q_u8(in_uv.ptr());
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_uv.val[0] = U0 U2 U4 U6 ...
        //ta_uv.val[1] = V0 V2 V4 V6 ...

        vst2q_u8(out_y.ptr(), ta_y_top);
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), ta_y_bottom);

        uint8x16x2_t uvec;
        uvec.val[0] = ta_uv.val[0 + shift];
        uvec.val[1] = ta_uv.val[0 + shift];
        vst2q_u8(out_u.ptr(), uvec);
        vst2q_u8(out_u.ptr() + output_ptr->plane(1)->info()->strides_in_bytes().y(), uvec);

        uint8x16x2_t vvec;
        vvec.val[0] = ta_uv.val[1 - shift];
        vvec.val[1] = ta_uv.val[1 - shift];
        vst2q_u8(out_v.ptr(), vvec);
        vst2q_u8(out_v.ptr() + output_ptr->plane(2)->info()->strides_in_bytes().y(), vvec);
    },
    in_y, in_uv, out_y, out_u, out_v);
}

/** Convert IYUV to YUV4.
 *
 * @param[in]  input  Input IYUV data buffer.
 * @param[out] output Output YUV4 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
void colorconvert_iyuv_to_yuv4(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IMultiImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in_y(input_ptr->plane(0), win);
    Iterator in_u(input_ptr->plane(1), win_uv);
    Iterator in_v(input_ptr->plane(2), win_uv);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win);
    Iterator out_v(output_ptr->plane(2), win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_y_top    = vld2q_u8(in_y.ptr());
        const auto ta_y_bottom = vld2q_u8(in_y.ptr() + input_ptr->plane(0)->info()->strides_in_bytes().y());
        const auto ta_u        = vld1q_u8(in_u.ptr());
        const auto ta_v        = vld1q_u8(in_v.ptr());
        //ta_y.val[0] = Y0 Y2 Y4 Y6 ...
        //ta_y.val[1] = Y1 Y3 Y5 Y7 ...
        //ta_u = U0 U2 U4 U6 ...
        //ta_v = V0 V2 V4 V6 ...

        vst2q_u8(out_y.ptr(), ta_y_top);
        vst2q_u8(out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(), ta_y_bottom);

        uint8x16x2_t uvec;
        uvec.val[0] = ta_u;
        uvec.val[1] = ta_u;
        vst2q_u8(out_u.ptr(), uvec);
        vst2q_u8(out_u.ptr() + output_ptr->plane(1)->info()->strides_in_bytes().y(), uvec);

        uint8x16x2_t vvec;
        vvec.val[0] = ta_v;
        vvec.val[1] = ta_v;
        vst2q_u8(out_v.ptr(), vvec);
        vst2q_u8(out_v.ptr() + output_ptr->plane(2)->info()->strides_in_bytes().y(), vvec);
    },
    in_y, in_u, in_v, out_y, out_u, out_v);
}

/** Convert RGB to NV12.
 *
 * @param[in]  input  Input RGB data buffer.
 * @param[out] output Output NV12 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool alpha>
void colorconvert_rgb_to_nv12(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in(input_ptr, win);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_uv(output_ptr->plane(1), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_rgb_top    = load_rgb(in.ptr(), alpha);
        const auto ta_rgb_bottom = load_rgb(in.ptr() + input_ptr->info()->strides_in_bytes().y(), alpha);
        //ta_rgb.val[0] = R0 R1 R2 R3 ...
        //ta_rgb.val[1] = G0 G1 G2 G3 ...
        //ta_rgb.val[2] = B0 B1 B2 B3 ...

        store_rgb_to_nv12(ta_rgb_top.val[0], ta_rgb_top.val[1], ta_rgb_top.val[2],
                          ta_rgb_bottom.val[0], ta_rgb_bottom.val[1], ta_rgb_bottom.val[2],
                          out_y.ptr(), out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(),
                          out_uv.ptr());
    },
    in, out_y, out_uv);
}

/** Convert RGB to IYUV.
 *
 * @param[in]  input  Input RGB data buffer.
 * @param[out] output Output IYUV buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool alpha>
void colorconvert_rgb_to_iyuv(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    // UV's width and height are subsampled
    Window win_uv(win);
    win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
    win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
    win_uv.validate();

    Iterator in(input_ptr, win);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win_uv);
    Iterator out_v(output_ptr->plane(2), win_uv);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_rgb_top    = load_rgb(in.ptr(), alpha);
        const auto ta_rgb_bottom = load_rgb(in.ptr() + input_ptr->info()->strides_in_bytes().y(), alpha);
        //ta_rgb.val[0] = R0 R1 R2 R3 ...
        //ta_rgb.val[1] = G0 G1 G2 G3 ...
        //ta_rgb.val[2] = B0 B1 B2 B3 ...

        store_rgb_to_iyuv(ta_rgb_top.val[0], ta_rgb_top.val[1], ta_rgb_top.val[2],
                          ta_rgb_bottom.val[0], ta_rgb_bottom.val[1], ta_rgb_bottom.val[2],
                          out_y.ptr(), out_y.ptr() + output_ptr->plane(0)->info()->strides_in_bytes().y(),
                          out_u.ptr(), out_v.ptr());
    },
    in, out_y, out_u, out_v);
}

/** Convert RGB to YUV4.
 *
 * @param[in]  input  Input RGB data buffer.
 * @param[out] output Output YUV4 buffer.
 * @param[in]  win    Window for iterating the buffers.
 *
 */
template <bool alpha>
void colorconvert_rgb_to_yuv4(const void *__restrict input, void *__restrict output, const Window &win)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    win.validate();

    const auto input_ptr  = static_cast<const IImage *__restrict>(input);
    const auto output_ptr = static_cast<IMultiImage *__restrict>(output);

    Iterator in(input_ptr, win);
    Iterator out_y(output_ptr->plane(0), win);
    Iterator out_u(output_ptr->plane(1), win);
    Iterator out_v(output_ptr->plane(2), win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto ta_rgb = load_rgb(in.ptr(), alpha);
        //ta_rgb.val[0] = R0 R1 R2 R3 ...
        //ta_rgb.val[1] = G0 G1 G2 G3 ...
        //ta_rgb.val[2] = B0 B1 B2 B3 ...

        store_rgb_to_yuv4(ta_rgb.val[0], ta_rgb.val[1], ta_rgb.val[2],
                          out_y.ptr(), out_u.ptr(), out_v.ptr());
    },
    in, out_y, out_u, out_v);
}
} // namespace arm_compute
