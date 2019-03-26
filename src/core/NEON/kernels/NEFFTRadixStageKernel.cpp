/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEFFTRadixStageKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cmath>
#include <complex>
#include <map>

#include "arm_compute/core/NEON/wrapper/traits.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace
{
// PI constant (from cmath)
constexpr float kPi = float(M_PI);

// Constant used in the fft_3 kernel
constexpr float kSqrt3Div2 = 0.866025403784438;

// Constants used in the fft_5 kernel
constexpr float kW5_0 = 0.30901699437494f;
constexpr float kW5_1 = 0.95105651629515f;
constexpr float kW5_2 = 0.80901699437494f;
constexpr float kW5_3 = 0.58778525229247f;

// Constants used in the fft_7 kernel
constexpr float kW7_0 = 0.62348980185873f;
constexpr float kW7_1 = 0.78183148246802f;
constexpr float kW7_2 = 0.22252093395631f;
constexpr float kW7_3 = 0.97492791218182f;
constexpr float kW7_4 = 0.90096886790241f;
constexpr float kW7_5 = 0.43388373911755f;

// Constant used in the fft_8 kernel
constexpr float kSqrt2Div2 = 0.707106781186548;

float32x2_t c_mul_neon(float32x2_t a, float32x2_t b)
{
    using ExactTagType = typename wrapper::traits::neon_vector<float, 2>::tag_type;

    const float32x2_t mask = { -1.0, 1.0 };
    const float32x2_t tmp0 = wrapper::vdup_n(wrapper::vgetlane(a, 0), ExactTagType{});
    const float32x2_t tmp1 = wrapper::vdup_n(wrapper::vgetlane(a, 1), ExactTagType{});

    float32x2_t res = wrapper::vmul(tmp0, b);

    b   = wrapper::vrev64(b);
    b   = wrapper::vmul(b, mask);
    res = wrapper::vmla(res, tmp1, b);

    return res;
}

float32x2_t c_mul_neon_img(float32x2_t a, float img_constant)
{
    const float a_r = wrapper::vgetlane(a, 0);
    const float a_i = wrapper::vgetlane(a, 1);

    const auto out = wrapper::vmul(float32x2_t{ -a_i, a_r }, float32x2_t{ img_constant, img_constant });
    return out;
}

float32x2_t reduce_sum_5(float32x2_t a, float32x2_t b, float32x2_t c, float32x2_t d, float32x2_t e)
{
    const auto t0 = wrapper::vadd(a, b);
    const auto t1 = wrapper::vadd(c, d);
    const auto t2 = wrapper::vadd(t0, t1);
    return wrapper::vadd(t2, e);
}

float32x2_t reduce_sum_7(float32x2_t x1, float32x2_t x2, float32x2_t x3, float32x2_t x4, float32x2_t x5, float32x2_t x6, float32x2_t x7)
{
    const auto t0  = wrapper::vadd(x1, x2);
    const auto t1  = wrapper::vadd(x3, x4);
    const auto t2  = wrapper::vadd(x5, x6);
    const auto t00 = wrapper::vadd(t0, t1);
    const auto t01 = wrapper::vadd(t2, x7);

    return wrapper::vadd(t00, t01);
}

float32x2_t reduce_sum_8(float32x2_t x1, float32x2_t x2, float32x2_t x3, float32x2_t x4, float32x2_t x5, float32x2_t x6, float32x2_t x7, float32x2_t x8)
{
    const auto t0  = wrapper::vadd(x1, x2);
    const auto t1  = wrapper::vadd(x3, x4);
    const auto t2  = wrapper::vadd(x5, x6);
    const auto t3  = wrapper::vadd(x7, x8);
    const auto t00 = wrapper::vadd(t0, t1);
    const auto t01 = wrapper::vadd(t2, t3);

    return wrapper::vadd(t00, t01);
}

void fft_2(float32x2_t &x, float32x2_t &y, float32x2_t &w)
{
    float32x2_t a = x;
    float32x2_t b = c_mul_neon(w, y);

    x = wrapper::vadd(a, b);
    y = wrapper::vsub(a, b);
}

void fft_3(float32x2_t &x, float32x2_t &y, float32x2_t &z, const float32x2_t &w, const float32x2_t &w2)
{
    float32x2_t a = x;
    float32x2_t b = c_mul_neon(w, y);
    float32x2_t c = c_mul_neon(w2, z);

    x = wrapper::vadd(a, b);
    x = wrapper::vadd(x, c);

    const auto v1 = wrapper::vmul(float32x2_t{ 0.5f, 0.5 }, wrapper::vadd(b, c));
    const auto v2 = c_mul_neon(float32x2_t{ 0.f, -kSqrt3Div2 }, wrapper::vsub(b, c));

    y = z = wrapper::vsub(a, v1);
    y     = wrapper::vadd(y, v2);
    z     = wrapper::vsub(z, v2);
}

void fft_4(float32x2_t &x1, float32x2_t &x2, float32x2_t &x3, float32x2_t &x4, const float32x2_t &w, const float32x2_t &w2, const float32x2_t &w3)
{
    float32x2_t a = x1;
    float32x2_t b = c_mul_neon(w, x2);
    float32x2_t c = c_mul_neon(w2, x3);
    float32x2_t d = c_mul_neon(w3, x4);

    const auto x11 = wrapper::vadd(a, b);
    const auto x12 = wrapper::vadd(c, d);
    x1             = wrapper::vadd(x11, x12);

    const auto x21 = wrapper::vadd(a, c_mul_neon_img(b, -1));
    const auto x22 = wrapper::vadd(wrapper::vneg(c), c_mul_neon_img(d, 1.f));
    x2             = wrapper::vadd(x21, x22);

    const auto x31 = wrapper::vadd(a, wrapper::vneg(b));
    const auto x32 = wrapper::vadd(c, wrapper::vneg(d));
    x3             = wrapper::vadd(x31, x32);

    const auto x41 = wrapper::vadd(a, c_mul_neon_img(b, 1));
    const auto x42 = wrapper::vadd(wrapper::vneg(c), c_mul_neon_img(d, -1));
    x4             = wrapper::vadd(x41, x42);
}

void fft_5(float32x2_t &x1, float32x2_t &x2, float32x2_t &x3, float32x2_t &x4, float32x2_t &x5, const float32x2_t &w, const float32x2_t &w2, const float32x2_t &w3, const float32x2_t &w4)
{
    const auto a = x1;
    const auto b = c_mul_neon(w, x2);
    const auto c = c_mul_neon(w2, x3);
    const auto d = c_mul_neon(w3, x4);
    const auto e = c_mul_neon(w4, x5);

    const auto b0 = c_mul_neon(float32x2_t{ kW5_0, -kW5_1 }, b);
    const auto b1 = c_mul_neon(float32x2_t{ -kW5_2, -kW5_3 }, b);
    const auto b2 = c_mul_neon(float32x2_t{ -kW5_2, kW5_3 }, b);
    const auto b3 = c_mul_neon(float32x2_t{ kW5_0, kW5_1 }, b);

    const auto c0 = c_mul_neon(float32x2_t{ -kW5_2, -kW5_3 }, c);
    const auto c1 = c_mul_neon(float32x2_t{ kW5_0, kW5_1 }, c);
    const auto c2 = c_mul_neon(float32x2_t{ kW5_0, -kW5_1 }, c);
    const auto c3 = c_mul_neon(float32x2_t{ -kW5_2, kW5_3 }, c);

    const auto d0 = c_mul_neon(float32x2_t{ -kW5_2, kW5_3 }, d);
    const auto d1 = c_mul_neon(float32x2_t{ kW5_0, -kW5_1 }, d);
    const auto d2 = c_mul_neon(float32x2_t{ kW5_0, kW5_1 }, d);
    const auto d3 = c_mul_neon(float32x2_t{ -kW5_2, -kW5_3 }, d);

    const auto e0 = c_mul_neon(float32x2_t{ kW5_0, kW5_1 }, e);
    const auto e1 = c_mul_neon(float32x2_t{ -kW5_2, kW5_3 }, e);
    const auto e2 = c_mul_neon(float32x2_t{ -kW5_2, -kW5_3 }, e);
    const auto e3 = c_mul_neon(float32x2_t{ kW5_0, -kW5_1 }, e);

    x1 = reduce_sum_5(a, b, c, d, e);
    x2 = reduce_sum_5(a, b0, c0, d0, e0);
    x3 = reduce_sum_5(a, b1, c1, d1, e1);
    x4 = reduce_sum_5(a, b2, c2, d2, e2);
    x5 = reduce_sum_5(a, b3, c3, d3, e3);
}

void fft_7(float32x2_t &x1, float32x2_t &x2, float32x2_t &x3, float32x2_t &x4, float32x2_t &x5, float32x2_t &x6, float32x2_t &x7, const float32x2_t &w, const float32x2_t &w2, const float32x2_t &w3,
           const float32x2_t &w4,
           const float32x2_t &w5, const float32x2_t &w6)
{
    const auto a = x1;
    const auto b = c_mul_neon(w, x2);
    const auto c = c_mul_neon(w2, x3);
    const auto d = c_mul_neon(w3, x4);
    const auto e = c_mul_neon(w4, x5);
    const auto f = c_mul_neon(w5, x6);
    const auto g = c_mul_neon(w6, x7);

    const auto b0 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, b);
    const auto b1 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, b);
    const auto b2 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, b);
    const auto b3 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, b);
    const auto b4 = c_mul_neon(float32x2_t{ -kW7_2, kW7_3 }, b);
    const auto b5 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, b);

    const auto c0 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, c);
    const auto c1 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, c);
    const auto c2 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, c);
    const auto c3 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, c);
    const auto c4 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, c);
    const auto c5 = c_mul_neon(float32x2_t{ -kW7_2, kW7_3 }, c);

    const auto d0 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, d);
    const auto d1 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, d);
    const auto d2 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, d);
    const auto d3 = c_mul_neon(float32x2_t{ -kW7_2, +kW7_3 }, d);
    const auto d4 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, d);
    const auto d5 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, d);

    const auto e0 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, e);
    const auto e1 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, e);
    const auto e2 = c_mul_neon(float32x2_t{ -kW7_2, kW7_3 }, e);
    const auto e3 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, e);
    const auto e4 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, e);
    const auto e5 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, e);

    const auto f0 = c_mul_neon(float32x2_t{ -kW7_2, kW7_3 }, f);
    const auto f1 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, f);
    const auto f2 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, f);
    const auto f3 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, f);
    const auto f4 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, f);
    const auto f5 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, f);

    const auto g0 = c_mul_neon(float32x2_t{ kW7_0, kW7_1 }, g);
    const auto g1 = c_mul_neon(float32x2_t{ -kW7_2, kW7_3 }, g);
    const auto g2 = c_mul_neon(float32x2_t{ -kW7_4, kW7_5 }, g);
    const auto g3 = c_mul_neon(float32x2_t{ -kW7_4, -kW7_5 }, g);
    const auto g4 = c_mul_neon(float32x2_t{ -kW7_2, -kW7_3 }, g);
    const auto g5 = c_mul_neon(float32x2_t{ kW7_0, -kW7_1 }, g);

    x1 = reduce_sum_7(a, b, c, d, e, f, g);
    x2 = reduce_sum_7(a, b0, c0, d0, e0, f0, g0);
    x3 = reduce_sum_7(a, b1, c1, d1, e1, f1, g1);
    x4 = reduce_sum_7(a, b2, c2, d2, e2, f2, g2);
    x5 = reduce_sum_7(a, b3, c3, d3, e3, f3, g3);
    x6 = reduce_sum_7(a, b4, c4, d4, e4, f4, g4);
    x7 = reduce_sum_7(a, b5, c5, d5, e5, f5, g5);
}

void fft_8(float32x2_t &x1, float32x2_t &x2, float32x2_t &x3, float32x2_t &x4, float32x2_t &x5, float32x2_t &x6, float32x2_t &x7, float32x2_t &x8, const float32x2_t &w, const float32x2_t &w2,
           const float32x2_t &w3,
           const float32x2_t &w4, const float32x2_t &w5, const float32x2_t &w6,
           const float32x2_t &w7)
{
    const auto a = x1;
    const auto b = c_mul_neon(w, x2);
    const auto c = c_mul_neon(w2, x3);
    const auto d = c_mul_neon(w3, x4);
    const auto e = c_mul_neon(w4, x5);
    const auto f = c_mul_neon(w5, x6);
    const auto g = c_mul_neon(w6, x7);
    const auto h = c_mul_neon(w7, x8);

    const auto b0 = c_mul_neon(float32x2_t{ kSqrt2Div2, -kSqrt2Div2 }, b);
    const auto b1 = c_mul_neon(float32x2_t{ 0, -1 }, b);
    const auto b2 = c_mul_neon(float32x2_t{ -kSqrt2Div2, -kSqrt2Div2 }, b);
    const auto b3 = c_mul_neon(float32x2_t{ -1, 0 }, b);
    const auto b4 = c_mul_neon(float32x2_t{ -kSqrt2Div2, kSqrt2Div2 }, b);
    const auto b5 = c_mul_neon(float32x2_t{ 0, 1 }, b);
    const auto b6 = c_mul_neon(float32x2_t{ kSqrt2Div2, kSqrt2Div2 }, b);

    const auto c0 = c_mul_neon(float32x2_t{ 0, -1 }, c);
    const auto c1 = c_mul_neon(float32x2_t{ -1, 0 }, c);
    const auto c2 = c_mul_neon(float32x2_t{ 0, 1 }, c);
    const auto c3 = c_mul_neon(float32x2_t{ 1, 0 }, c);
    const auto c4 = c_mul_neon(float32x2_t{ 0, -1 }, c);
    const auto c5 = c_mul_neon(float32x2_t{ -1, 0 }, c);
    const auto c6 = c_mul_neon(float32x2_t{ 0, 1 }, c);

    const auto d0 = c_mul_neon(float32x2_t{ -kSqrt2Div2, -kSqrt2Div2 }, d);
    const auto d1 = c_mul_neon(float32x2_t{ 0, 1 }, d);
    const auto d2 = c_mul_neon(float32x2_t{ kSqrt2Div2, -kSqrt2Div2 }, d);
    const auto d3 = c_mul_neon(float32x2_t{ -1, 0 }, d);
    const auto d4 = c_mul_neon(float32x2_t{ kSqrt2Div2, kSqrt2Div2 }, d);
    const auto d5 = c_mul_neon(float32x2_t{ 0, -1 }, d);
    const auto d6 = c_mul_neon(float32x2_t{ -kSqrt2Div2, kSqrt2Div2 }, d);

    const auto e0 = c_mul_neon(float32x2_t{ -1, 0 }, e);
    const auto e1 = c_mul_neon(float32x2_t{ 1, 0 }, e);
    const auto e2 = c_mul_neon(float32x2_t{ -1, 0 }, e);
    const auto e3 = c_mul_neon(float32x2_t{ 1, 0 }, e);
    const auto e4 = c_mul_neon(float32x2_t{ -1, 0 }, e);
    const auto e5 = c_mul_neon(float32x2_t{ 1, 0 }, e);
    const auto e6 = c_mul_neon(float32x2_t{ -1, 0 }, e);

    const auto f0 = c_mul_neon(float32x2_t{ -kSqrt2Div2, kSqrt2Div2 }, f);
    const auto f1 = c_mul_neon(float32x2_t{ 0, -1 }, f);
    const auto f2 = c_mul_neon(float32x2_t{ kSqrt2Div2, kSqrt2Div2 }, f);
    const auto f3 = c_mul_neon(float32x2_t{ -1, 0 }, f);
    const auto f4 = c_mul_neon(float32x2_t{ kSqrt2Div2, -kSqrt2Div2 }, f);
    const auto f5 = c_mul_neon(float32x2_t{ 0, 1 }, f);
    const auto f6 = c_mul_neon(float32x2_t{ -kSqrt2Div2, -kSqrt2Div2 }, f);

    const auto g0 = c_mul_neon(float32x2_t{ 0, 1 }, g);
    const auto g1 = c_mul_neon(float32x2_t{ -1, 0 }, g);
    const auto g2 = c_mul_neon(float32x2_t{ 0, -1 }, g);
    const auto g3 = c_mul_neon(float32x2_t{ 1, 0 }, g);
    const auto g4 = c_mul_neon(float32x2_t{ 0, 1 }, g);
    const auto g5 = c_mul_neon(float32x2_t{ -1, 0 }, g);
    const auto g6 = c_mul_neon(float32x2_t{ 0, -1 }, g);

    const auto h0 = c_mul_neon(float32x2_t{ kSqrt2Div2, kSqrt2Div2 }, h);
    const auto h1 = c_mul_neon(float32x2_t{ 0, 1 }, h);
    const auto h2 = c_mul_neon(float32x2_t{ -kSqrt2Div2, kSqrt2Div2 }, h);
    const auto h3 = c_mul_neon(float32x2_t{ -1, 0 }, h);
    const auto h4 = c_mul_neon(float32x2_t{ -kSqrt2Div2, -kSqrt2Div2 }, h);
    const auto h5 = c_mul_neon(float32x2_t{ 0, -1 }, h);
    const auto h6 = c_mul_neon(float32x2_t{ kSqrt2Div2, -kSqrt2Div2 }, h);

    x1 = reduce_sum_8(a, b, c, d, e, f, g, h);
    x2 = reduce_sum_8(a, b0, c0, d0, e0, f0, g0, h0);
    x3 = reduce_sum_8(a, b1, c1, d1, e1, f1, g1, h1);
    x4 = reduce_sum_8(a, b2, c2, d2, e2, f2, g2, h2);
    x5 = reduce_sum_8(a, b3, c3, d3, e3, f3, g3, h3);
    x6 = reduce_sum_8(a, b4, c4, d4, e4, f4, g4, h4);
    x7 = reduce_sum_8(a, b5, c5, d5, e5, f5, g5, h5);
    x8 = reduce_sum_8(a, b6, c6, d6, e6, f6, g6, h6);
}

template <bool first_stage>
void fft_radix_2_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            auto a = float32x2_t{ 0, 0 };
            auto b = float32x2_t{ 0, 0 };

            // Load inputs
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                a             = wrapper::vgetlow(ab);
                b             = wrapper::vgethigh(ab);
            }
            else
            {
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
            }

            // Base-case prime transform
            fft_2(a, b, w);

            // Write outputs
            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
            }
        }

        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_2_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));

            // Base-case prime transform
            fft_2(a, b, w);

            // Write outputs
            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
        }

        w = c_mul_neon(w, w_m);
    }
}

template <bool first_stage>
void fft_radix_3_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const auto w2 = c_mul_neon(w, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = { 0, 0 };
            float32x2_t b = { 0, 0 };
            float32x2_t c = { 0, 0 };
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                a             = wrapper::vgetlow(ab);
                b             = wrapper::vgethigh(ab);
            }
            else
            {
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
            }
            c = wrapper::vload(x + k + 4 * Nx);

            // Base-case prime transform
            fft_3(a, b, c, w, w2);

            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
            }
            wrapper::vstore(X + k + 4 * Nx, c);
        }
        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_3_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const auto w2 = c_mul_neon(w, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));
            float32x2_t c = wrapper::vload(x + M * (k + 4 * Nx));

            // Base-case prime transform
            fft_3(a, b, c, w, w2);

            // Store the output
            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
            wrapper::vstore(X + M * (k + 4 * Nx), c);
        }
        w = c_mul_neon(w, w_m);
    }
}

template <bool first_stage>
void fft_radix_4_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const auto w2 = c_mul_neon(w, w);
        const auto w3 = c_mul_neon(w2, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            float32x2_t a = { 0, 0 };
            float32x2_t b = { 0, 0 };
            float32x2_t c = { 0, 0 };
            float32x2_t d = { 0, 0 };
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                const auto cd = wrapper::vloadq(x + k + 4 * Nx);
                a             = wrapper::vgetlow(ab);
                b             = wrapper::vgethigh(ab);
                c             = wrapper::vgetlow(cd);
                d             = wrapper::vgethigh(cd);
            }
            else
            {
                // Load inputs
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
                c = wrapper::vload(x + k + 4 * Nx);
                d = wrapper::vload(x + k + 6 * Nx);
            }

            // Base-case prime transform
            fft_4(a, b, c, d, w, w2, w3);

            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
                wrapper::vstore(X + k + 4 * Nx, wrapper::vcombine(c, d));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
                wrapper::vstore(X + k + 4 * Nx, c);
                wrapper::vstore(X + k + 6 * Nx, d);
            }
        }

        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_4_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const auto w2 = c_mul_neon(w, w);
        const auto w3 = c_mul_neon(w2, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));
            float32x2_t c = wrapper::vload(x + M * (k + 4 * Nx));
            float32x2_t d = wrapper::vload(x + M * (k + 6 * Nx));

            // Base-case prime transform
            fft_4(a, b, c, d, w, w2, w3);

            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
            wrapper::vstore(X + M * (k + 4 * Nx), c);
            wrapper::vstore(X + M * (k + 6 * Nx), d);
        }

        w = c_mul_neon(w, w_m);
    }
}

template <bool first_stage>
void fft_radix_5_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            float32x2_t a = { 0, 0 };
            float32x2_t b = { 0, 0 };
            float32x2_t c = { 0, 0 };
            float32x2_t d = { 0, 0 };
            float32x2_t e = { 0, 0 };

            // Load inputs
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                const auto cd = wrapper::vloadq(x + k + 4 * Nx);

                a = wrapper::vgetlow(ab);
                b = wrapper::vgethigh(ab);
                c = wrapper::vgetlow(cd);
                d = wrapper::vgethigh(cd);
            }
            else
            {
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
                c = wrapper::vload(x + k + 4 * Nx);
                d = wrapper::vload(x + k + 6 * Nx);
            }
            e = wrapper::vload(x + k + 8 * Nx);

            // Base-case prime transform
            fft_5(a, b, c, d, e, w, w2, w3, w4);

            // Store outputs
            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
                wrapper::vstore(X + k + 4 * Nx, wrapper::vcombine(c, d));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
                wrapper::vstore(X + k + 4 * Nx, c);
                wrapper::vstore(X + k + 6 * Nx, d);
            }
            wrapper::vstore(X + k + 8 * Nx, e);
        }

        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_5_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));
            float32x2_t c = wrapper::vload(x + M * (k + 4 * Nx));
            float32x2_t d = wrapper::vload(x + M * (k + 6 * Nx));
            float32x2_t e = wrapper::vload(x + M * (k + 8 * Nx));

            // Base-case prime transform
            fft_5(a, b, c, d, e, w, w2, w3, w4);

            // Store outputs
            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
            wrapper::vstore(X + M * (k + 4 * Nx), c);
            wrapper::vstore(X + M * (k + 6 * Nx), d);
            wrapper::vstore(X + M * (k + 8 * Nx), e);
        }

        w = c_mul_neon(w, w_m);
    }
}

template <bool first_stage>
void fft_radix_7_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);
        const float32x2_t w5 = c_mul_neon(w4, w);
        const float32x2_t w6 = c_mul_neon(w5, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            float32x2_t a = { 0, 0 };
            float32x2_t b = { 0, 0 };
            float32x2_t c = { 0, 0 };
            float32x2_t d = { 0, 0 };
            float32x2_t e = { 0, 0 };
            float32x2_t f = { 0, 0 };
            float32x2_t g = { 0, 0 };

            // Load inputs
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                const auto cd = wrapper::vloadq(x + k + 4 * Nx);
                const auto ef = wrapper::vloadq(x + k + 8 * Nx);

                a = wrapper::vgetlow(ab);
                b = wrapper::vgethigh(ab);
                c = wrapper::vgetlow(cd);
                d = wrapper::vgethigh(cd);
                e = wrapper::vgetlow(ef);
                f = wrapper::vgethigh(ef);
            }
            else
            {
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
                c = wrapper::vload(x + k + 4 * Nx);
                d = wrapper::vload(x + k + 6 * Nx);
                e = wrapper::vload(x + k + 8 * Nx);
                f = wrapper::vload(x + k + 10 * Nx);
            }
            g = wrapper::vload(x + k + 12 * Nx);

            // Base-case prime transform
            fft_7(a, b, c, d, e, f, g, w, w2, w3, w4, w5, w6);

            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
                wrapper::vstore(X + k + 4 * Nx, wrapper::vcombine(c, d));
                wrapper::vstore(X + k + 8 * Nx, wrapper::vcombine(e, f));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
                wrapper::vstore(X + k + 4 * Nx, c);
                wrapper::vstore(X + k + 6 * Nx, d);
                wrapper::vstore(X + k + 8 * Nx, e);
                wrapper::vstore(X + k + 10 * Nx, f);
            }
            wrapper::vstore(X + k + 12 * Nx, g);
        }

        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_7_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);
        const float32x2_t w5 = c_mul_neon(w4, w);
        const float32x2_t w6 = c_mul_neon(w5, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));
            float32x2_t c = wrapper::vload(x + M * (k + 4 * Nx));
            float32x2_t d = wrapper::vload(x + M * (k + 6 * Nx));
            float32x2_t e = wrapper::vload(x + M * (k + 8 * Nx));
            float32x2_t f = wrapper::vload(x + M * (k + 10 * Nx));
            float32x2_t g = wrapper::vload(x + M * (k + 12 * Nx));

            // Base-case prime transform
            fft_7(a, b, c, d, e, f, g, w, w2, w3, w4, w5, w6);

            // Store outputs
            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
            wrapper::vstore(X + M * (k + 4 * Nx), c);
            wrapper::vstore(X + M * (k + 6 * Nx), d);
            wrapper::vstore(X + M * (k + 8 * Nx), e);
            wrapper::vstore(X + M * (k + 10 * Nx), f);
            wrapper::vstore(X + M * (k + 12 * Nx), g);
        }

        w = c_mul_neon(w, w_m);
    }
}

template <bool first_stage>
void fft_radix_8_axes_0(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);
        const float32x2_t w5 = c_mul_neon(w4, w);
        const float32x2_t w6 = c_mul_neon(w5, w);
        const float32x2_t w7 = c_mul_neon(w6, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = { 0, 0 };
            float32x2_t b = { 0, 0 };
            float32x2_t c = { 0, 0 };
            float32x2_t d = { 0, 0 };
            float32x2_t e = { 0, 0 };
            float32x2_t f = { 0, 0 };
            float32x2_t g = { 0, 0 };
            float32x2_t h = { 0, 0 };

            // Base-case prime transform
            if(first_stage)
            {
                const auto ab = wrapper::vloadq(x + k);
                const auto cd = wrapper::vloadq(x + k + 4 * Nx);
                const auto ef = wrapper::vloadq(x + k + 8 * Nx);
                const auto gh = wrapper::vloadq(x + k + 12 * Nx);

                a = wrapper::vgetlow(ab);
                b = wrapper::vgethigh(ab);
                c = wrapper::vgetlow(cd);
                d = wrapper::vgethigh(cd);
                e = wrapper::vgetlow(ef);
                f = wrapper::vgethigh(ef);
                g = wrapper::vgetlow(gh);
                h = wrapper::vgethigh(gh);
            }
            else
            {
                a = wrapper::vload(x + k);
                b = wrapper::vload(x + k + 2 * Nx);
                c = wrapper::vload(x + k + 4 * Nx);
                d = wrapper::vload(x + k + 6 * Nx);
                e = wrapper::vload(x + k + 8 * Nx);
                f = wrapper::vload(x + k + 10 * Nx);
                g = wrapper::vload(x + k + 12 * Nx);
                h = wrapper::vload(x + k + 14 * Nx);
            }

            // Apply twiddle factors
            fft_8(a, b, c, d, e, f, g, h, w, w2, w3, w4, w5, w6, w7);

            // Store outputs
            if(first_stage)
            {
                wrapper::vstore(X + k, wrapper::vcombine(a, b));
                wrapper::vstore(X + k + 4 * Nx, wrapper::vcombine(c, d));
                wrapper::vstore(X + k + 8 * Nx, wrapper::vcombine(e, f));
                wrapper::vstore(X + k + 12 * Nx, wrapper::vcombine(g, h));
            }
            else
            {
                wrapper::vstore(X + k, a);
                wrapper::vstore(X + k + 2 * Nx, b);
                wrapper::vstore(X + k + 4 * Nx, c);
                wrapper::vstore(X + k + 6 * Nx, d);
                wrapper::vstore(X + k + 8 * Nx, e);
                wrapper::vstore(X + k + 10 * Nx, f);
                wrapper::vstore(X + k + 12 * Nx, g);
                wrapper::vstore(X + k + 14 * Nx, h);
            }
        }

        w = c_mul_neon(w, w_m);
    }
}

void fft_radix_8_axes_1(float *X, float *x, unsigned int Nx, unsigned int NxRadix, const float32x2_t &w_m, unsigned int M, unsigned int N)
{
    float32x2_t w{ 1.0f, 0.0f };
    for(unsigned int j = 0; j < Nx; j++)
    {
        const float32x2_t w2 = c_mul_neon(w, w);
        const float32x2_t w3 = c_mul_neon(w2, w);
        const float32x2_t w4 = c_mul_neon(w3, w);
        const float32x2_t w5 = c_mul_neon(w4, w);
        const float32x2_t w6 = c_mul_neon(w5, w);
        const float32x2_t w7 = c_mul_neon(w6, w);

        for(unsigned int k = 2 * j; k < 2 * N; k += 2 * NxRadix)
        {
            // Load inputs
            float32x2_t a = wrapper::vload(x + M * k);
            float32x2_t b = wrapper::vload(x + M * (k + 2 * Nx));
            float32x2_t c = wrapper::vload(x + M * (k + 4 * Nx));
            float32x2_t d = wrapper::vload(x + M * (k + 6 * Nx));
            float32x2_t e = wrapper::vload(x + M * (k + 8 * Nx));
            float32x2_t f = wrapper::vload(x + M * (k + 10 * Nx));
            float32x2_t g = wrapper::vload(x + M * (k + 12 * Nx));
            float32x2_t h = wrapper::vload(x + M * (k + 14 * Nx));

            // Base-case prime transform
            fft_8(a, b, c, d, e, f, g, h, w, w2, w3, w4, w5, w6, w7);

            // Store outputs
            wrapper::vstore(X + M * k, a);
            wrapper::vstore(X + M * (k + 2 * Nx), b);
            wrapper::vstore(X + M * (k + 4 * Nx), c);
            wrapper::vstore(X + M * (k + 6 * Nx), d);
            wrapper::vstore(X + M * (k + 8 * Nx), e);
            wrapper::vstore(X + M * (k + 10 * Nx), f);
            wrapper::vstore(X + M * (k + 12 * Nx), g);
            wrapper::vstore(X + M * (k + 14 * Nx), h);
        }

        w = c_mul_neon(w, w_m);
    }
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const FFTRadixStageKernelInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(config.axis > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(NEFFTRadixStageKernel::supported_radix().count(config.radix) == 0);
    ARM_COMPUTE_UNUSED(config);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const FFTRadixStageKernelInfo &config)
{
    ARM_COMPUTE_UNUSED(config);

    if(output != nullptr)
    {
        auto_init_if_empty(*output, *input);
    }

    Window win = calculate_max_window(*input, Steps());
    if(output != nullptr)
    {
        output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

NEFFTRadixStageKernel::NEFFTRadixStageKernel()
    : _input(nullptr), _output(nullptr), _run_in_place(false), _Nx(0), _axis(0), _radix(0), _func_0(), _func_1()
{
}

void NEFFTRadixStageKernel::set_radix_stage_axis0(const FFTRadixStageKernelInfo &config)
{
    // FFT table axis 0: [radix, first_stage]
    static std::map<unsigned int, std::map<bool, FFTFunctionPointerAxis0>> fft_table_axis0;

    if(fft_table_axis0.empty())
    {
        fft_table_axis0[2][false] = &fft_radix_2_axes_0<false>;
        fft_table_axis0[3][false] = &fft_radix_3_axes_0<false>;
        fft_table_axis0[4][false] = &fft_radix_4_axes_0<false>;
        fft_table_axis0[5][false] = &fft_radix_5_axes_0<false>;
        fft_table_axis0[7][false] = &fft_radix_7_axes_0<false>;
        fft_table_axis0[8][false] = &fft_radix_8_axes_0<false>;

        fft_table_axis0[2][true] = &fft_radix_2_axes_0<true>;
        fft_table_axis0[3][true] = &fft_radix_3_axes_0<true>;
        fft_table_axis0[4][true] = &fft_radix_4_axes_0<true>;
        fft_table_axis0[5][true] = &fft_radix_5_axes_0<true>;
        fft_table_axis0[7][true] = &fft_radix_7_axes_0<true>;
        fft_table_axis0[8][true] = &fft_radix_8_axes_0<true>;
    }

    _func_0 = fft_table_axis0[config.radix][config.is_first_stage];
}

void NEFFTRadixStageKernel::set_radix_stage_axis1(const FFTRadixStageKernelInfo &config)
{
    // FFT table axis 1: [radix, first_stage]
    static std::map<unsigned int, FFTFunctionPointerAxis1> fft_table_axis1;

    if(fft_table_axis1.empty())
    {
        fft_table_axis1[2] = &fft_radix_2_axes_1;
        fft_table_axis1[3] = &fft_radix_3_axes_1;
        fft_table_axis1[4] = &fft_radix_4_axes_1;
        fft_table_axis1[5] = &fft_radix_5_axes_1;
        fft_table_axis1[7] = &fft_radix_7_axes_1;
        fft_table_axis1[8] = &fft_radix_8_axes_1;
    }

    _func_1 = fft_table_axis1[config.radix];
}

void NEFFTRadixStageKernel::configure(ITensor *input, ITensor *output, const FFTRadixStageKernelInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    // Output auto inizialitation if not yet initialized
    if(output != nullptr)
    {
        auto_init_if_empty(*output->info(), *input->info()->clone());
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr, config));

    _input        = input;
    _output       = output;
    _run_in_place = (output == nullptr) || (output == input);
    _Nx           = config.Nx;
    _axis         = config.axis;
    _radix        = config.radix;

    switch(config.axis)
    {
        case 0:
            set_radix_stage_axis0(config);
            break;
        case 1:
            set_radix_stage_axis1(config);
            break;
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
            break;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (_run_in_place) ? nullptr : output->info(), config);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEFFTRadixStageKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const FFTRadixStageKernelInfo &config)
{
    const bool run_in_place = (output == nullptr) || (output == input);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, config));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              (run_in_place) ? nullptr : output->clone().get(),
                                                              config)
                                .first);

    return Status{};
}

std::set<unsigned int> NEFFTRadixStageKernel::supported_radix()
{
    return std::set<unsigned int> { 2, 3, 4, 5, 7, 8 };
}

void NEFFTRadixStageKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_UNUSED(info);

    Window input_window = window;
    input_window.set(_axis, 0);

    Iterator in(_input, input_window);
    Iterator out(_run_in_place ? _input : _output, input_window);

    // Precompute FFT constants
    const unsigned int NxRadix = _radix * _Nx;
    const float        alpha   = 2.0f * kPi / float(NxRadix);
    const float32x2_t  w_m{ cosf(alpha), -sinf(alpha) };

    if(_axis == 0)
    {
        const unsigned int N = _input->info()->dimension(0);
        execute_window_loop(input_window, [&](const Coordinates &)
        {
            _func_0(reinterpret_cast<float *>(out.ptr()), reinterpret_cast<float *>(in.ptr()), _Nx, NxRadix, w_m, N);
        },
        in, out);
    }
    else
    {
        const unsigned int N = _input->info()->dimension(0);
        const unsigned int M = _input->info()->dimension(1);
        execute_window_loop(input_window, [&](const Coordinates &)
        {
            _func_1(reinterpret_cast<float *>(out.ptr()), reinterpret_cast<float *>(in.ptr()), _Nx, NxRadix, w_m, N, M);
        },
        in, out);
    }

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
}
} // namespace arm_compute
