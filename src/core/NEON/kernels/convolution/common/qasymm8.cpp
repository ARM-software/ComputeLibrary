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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

#include "arm_compute/core/NEON/kernels/convolution/common/qasymm8.hpp"

namespace qasymm8
{
#if(__ANDROID__ || BARE_METAL)
template <typename T> T round(T val) {  return ::round(val); }
template <typename T> T exp2(T val) { return ::exp2(val); }
template <typename T> T log2(T val) { return ::log2(val); }
#else  /* (__ANDROID__ || BARE_METAL) */
template <typename T> T round(T val) { return std::round(val); }
template <typename T> T exp2(T val) { return std::exp2(val); }
template <typename T> T log2(T val) { return std::log2(val); }
#endif  /* (__ANDROID__ || BARE_METAL) */

uint8_t QAsymm8Params::quantize(const float value) const
{
  const float transformed = value / scale + offset;
  return static_cast<uint8_t>(round(std::max(0.0f, std::min(255.0f, transformed))));
}

float QAsymm8Params::dequantize(const uint8_t value) const
{
  return scale * (static_cast<float>(value) - offset);
}

QAsymm8RescaleParams QAsymm8RescaleParams::make_rescale_params(
  const QAsymm8Params& weight_quant,
  const QAsymm8Params& input_quant,
  const QAsymm8Params& output_quant
)
{
  // Based on the gemmlowp approach: https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
  const float rescale = weight_quant.scale * input_quant.scale / output_quant.scale;
  const float shiftf = round(log2(0.5f / rescale));
  const float multf = exp2(31.0f + shiftf)*rescale;

  int64_t shift = static_cast<int64_t>(shiftf);
  int64_t mult = static_cast<int64_t>(multf);

  if (mult == (1ll << 31))
  {
    mult /= 2;
    shift--;
  }

  assert(shift >= 0);
  assert(mult <= std::numeric_limits<int32_t>::max());

  return QAsymm8RescaleParams(
    static_cast<int32_t>(shift),
    static_cast<int32_t>(mult),
    rescale
  );
}

QAsymm8RescaleParams::QAsymm8RescaleParams(int32_t shift, int32_t multi, float rescale)
  : shift(shift), multiplier(multi), rescale(rescale)
{
}
}
