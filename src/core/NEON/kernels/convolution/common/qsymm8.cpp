/*
 * Copyright (c) 2019 Arm Limited.
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

#include "qsymm8.hpp"

namespace qsymm8 {
#if(__ANDROID__ || BARE_METAL)
template <typename T> T round(T val) {  return ::round(val); }
template <typename T> T exp2(T val) { return ::exp2(val); }
template <typename T> T log2(T val) { return ::log2(val); }
#else  /* (__ANDROID__ || BARE_METAL) */
template <typename T> T round(T val) { return std::round(val); }
template <typename T> T exp2(T val) { return std::exp2(val); }
template <typename T> T log2(T val) { return std::log2(val); }
#endif  /* (__ANDROID__ || BARE_METAL) */

// Symmetric quantization
int8_t QSymm8Params::quantize(float value) const
{
  const float transformed = value / scale;
  return static_cast<int8_t>(round(std::max(-128.0f, std::min(127.0f, transformed))));
}

float QSymm8Params::dequantize(const int8_t value) const
{
  return scale * (static_cast<float>(value));
}

QSymm8RescaleParams QSymm8RescaleParams::make_rescale_params(
  const QSymm8Params& weight_quant,
  const QSymm8Params& input_quant,
  const QSymm8Params& output_quant
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

  return QSymm8RescaleParams(
    static_cast<int32_t>(shift),
    static_cast<int32_t>(mult),
    rescale
  );
}

QSymm8RescaleParams::QSymm8RescaleParams(int32_t shift, int32_t multi, float rescale)
  : shift(shift), multiplier(multi), rescale(rescale)
{
}

// Symmetric per-channel quantization
int8_t QSymm8PerChannelParams::quantize(float value, float scale) const
{
  const float transformed = value / scale;
  return static_cast<int8_t>(round(std::max(-128.0f, std::min(127.0f, transformed))));
}

float QSymm8PerChannelParams::dequantize(const int8_t value, float scale) const
{
  return scale * (static_cast<float>(value));
}

QSymm8PerChannelRescaleParams QSymm8PerChannelRescaleParams::make_rescale_params(
  const QSymm8PerChannelParams& weight_quant,
  const QSymm8PerChannelParams& input_quant,
  const QSymm8PerChannelParams& output_quant
)
{
    std::vector<int32_t> shifts;
    std::vector<int32_t> mults;
    std::vector<float> rescales;

    for(size_t s = 0; s< input_quant.scales.size(); s++)
    {
          // Based on the gemmlowp approach: https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
          const float rescale = weight_quant.scales[s] * input_quant.scales[s] / output_quant.scales[s];
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

          shifts.push_back(static_cast<int32_t>(shift));
          mults.push_back(static_cast<int32_t>(mult));
          rescales.push_back(rescale);
    }

  return QSymm8PerChannelRescaleParams(shifts, mults, rescales);

}

QSymm8PerChannelRescaleParams QSymm8PerChannelRescaleParams::make_rescale_params(
  const QSymm8PerChannelParams& weight_quant,
  const qasymm8::QAsymm8Params& input_quant,
  const qasymm8::QAsymm8Params& output_quant
)
{
    std::vector<int32_t> shifts;
    std::vector<int32_t> mults;
    std::vector<float> rescales;

    for(size_t s = 0; s< weight_quant.scales.size(); s++)
    {
          // Based on the gemmlowp approach: https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
          const float rescale = weight_quant.scales[s] * input_quant.scale / output_quant.scale;
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

          shifts.push_back(static_cast<int32_t>(shift));
          mults.push_back(static_cast<int32_t>(mult));
          rescales.push_back(rescale);
    }

  return QSymm8PerChannelRescaleParams(shifts, mults, rescales);

}

QSymm8PerChannelRescaleParams::QSymm8PerChannelRescaleParams(std::vector<int32_t>& shifts, std::vector<int32_t>& multipliers, std::vector<float>& rescales)
  : shifts(shifts), multipliers(multipliers), rescales(rescales)
{
}


} // namespace qasymm8
