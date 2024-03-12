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

#pragma once
#include <cstdint>
#include <vector>
#include "qasymm8.hpp"


namespace qsymm8 {

struct QSymm8Params {
  int8_t quantize(float value) const;
  float dequantize(int8_t value) const;

  float scale;
};

struct QSymm8RescaleParams {
  static QSymm8RescaleParams
  make_rescale_params(const QSymm8Params &weight_quant,
                      const QSymm8Params &input_quant,
                      const QSymm8Params &output_quant);

  QSymm8RescaleParams(int32_t shift, int32_t multiplier, float rescale);

  const int32_t shift, multiplier;
  const float rescale;
};

struct QSymm8PerChannelParams {
  int8_t quantize(float value, float scale) const;
  float dequantize(int8_t value, float scale) const;

  std::vector<float> scales;
};

struct QSymm8PerChannelRescaleParams {
  static QSymm8PerChannelRescaleParams
  make_rescale_params(const QSymm8PerChannelParams &weight_quant,
                      const QSymm8PerChannelParams &input_quant,
                      const QSymm8PerChannelParams &output_quant);

  static QSymm8PerChannelRescaleParams
  make_rescale_params(const QSymm8PerChannelParams &weight_quant,
                      const qasymm8::QAsymm8Params &input_quant,
                      const qasymm8::QAsymm8Params &output_quant);

  QSymm8PerChannelRescaleParams(std::vector<int32_t>& shift, std::vector<int32_t>& multiplier, std::vector<float>& rescale);

  std::vector<int32_t>  shifts, multipliers;
  std::vector<float> rescales;
};

} // namespace qsymm8
