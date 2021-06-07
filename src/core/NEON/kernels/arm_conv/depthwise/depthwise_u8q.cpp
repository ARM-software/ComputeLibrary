/*
 * Copyright (c) 2021 Arm Limited.
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

#include "arm_gemm_local.hpp"

#include "depthwise_implementation.hpp"
#include "depthwise_depthfirst_quantized.hpp"
#include "depthwise_depthfirst_generic_quantized.hpp"
#include "depthwise_depthfirst_multiplier_quantized.hpp"
#include "depthwise_depthfirst_generic_multiplier_quantized.hpp"

#include "depthwise_implementation_constraints.hpp"

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
#include "kernels/sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst.hpp"
#include "kernels/sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
#include "kernels/a64_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_u8q_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/a64_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst.hpp"
#include "kernels/a64_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst.hpp"
#include "kernels/a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
#endif  // defined(__aarch64__)

#include <cstdint>

using arm_gemm::Requantize32;

namespace arm_conv {
namespace depthwise {

static const DepthwiseImplementation<uint8_t, uint8_t, uint8_t, Requantize32> depthwise_u8q_methods[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<sve_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<sve_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstWithMultiplierQuantized<sve_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstWithMultiplierQuantized<sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>(args, qp);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             cpu_has_dot_product,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<a64_u8q_nhwc_3x3_s1_output2x2_dot_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<a64_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstQuantized<a64_u8q_nhwc_5x5_s1_output2x2_mla_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_nhwc_generic_output3x3_mla_depthfirst",
    constraint<Requantize32>(has_no_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstGenericQuantized<a64_u8q_nhwc_generic_output9_mla_depthfirst, 3, 3>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>,
                             cpu_has_dot_product,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstWithMultiplierQuantized<a64_u8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>,
                             cpu_has_dot_product,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstWithMultiplierQuantized<a64_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>(args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    nullptr,
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<uint8_t, uint8_t, uint8_t> * {
      return new DepthwiseDepthfirstGenericWithMultiplierQuantized<a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst>(args, qp);
    },
  },
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<uint8_t, uint8_t, uint8_t, Requantize32> *depthwise_implementation_list()
{
  return depthwise_u8q_methods;
}

template UniqueDepthwiseCommon<uint8_t, uint8_t, uint8_t> depthwise(const DepthwiseArgs &, const Requantize32 &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint8_t, uint8_t, Requantize32>(const DepthwiseArgs &, const Requantize32 &);

}  // namespace depthwise
}  // namespace arm_conv
