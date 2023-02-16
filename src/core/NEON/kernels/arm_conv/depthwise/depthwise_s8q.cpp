/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "depthwise_depthfirst.hpp"
#include "depthwise_depthfirst_generic.hpp"
#include "depthwise_depthfirst_multiplier.hpp"
#include "depthwise_planar.hpp"

#include "depthwise_implementation_constraints.hpp"

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(ARM_COMPUTE_ENABLE_SME2)
#include "kernels/sme2_s8q_planar_3x3_s1_4rows_dot_za.hpp"
#include "kernels/sme2_s8q_planar_3x3_s2_4rows_dot_za.hpp"
#include "kernels/sme2_s8q_planar_5x5_s1_4rows_dot_za.hpp"
#include "kernels/sme2_s8q_planar_5x5_s2_4rows_dot_za.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SME2)

#include "kernels/sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst.hpp"
#include "kernels/sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/a64_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst.hpp"
#include "kernels/a64_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst.hpp"
#include "kernels/a64_s8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
#endif  // defined(__aarch64__)

#include <cstdint>

using arm_gemm::Requantize32;

namespace arm_conv {
namespace depthwise {

namespace
{
#if defined(__aarch64__)
bool qp_weights_are_symmetric(const DepthwiseArgs &, const void *_qp)
{
  const auto qp = static_cast<const arm_gemm::Requantize32 *>(_qp);
  return qp->b_offset == 0;
}
#endif // defined(__aarch64__)
}

static const DepthwiseImplementation<int8_t, int8_t, int8_t, Requantize32> depthwise_s8q_methods[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(ARM_COMPUTE_ENABLE_SME2)
  {
    DepthwiseMethod::PLANAR,
    "sme2_s8q_planar_3x3_s1_4rows_dot_za",
    constraint<Requantize32>(cpu_has_sme, cpu_has_sme2,
                             is_supported<sme2_s8q_planar_3x3_s1_4rows_dot_za>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sme2_s8q_planar_3x3_s1_4rows_dot_za(args.cpu_info);
      return new DepthwisePlanar<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_s8q_planar_3x3_s2_4rows_dot_za",
    constraint<Requantize32>(cpu_has_sme, cpu_has_sme2,
                             is_supported<sme2_s8q_planar_3x3_s2_4rows_dot_za>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sme2_s8q_planar_3x3_s2_4rows_dot_za(args.cpu_info);
      return new DepthwisePlanar<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_s8q_planar_5x5_s1_4rows_dot_za",
    constraint<Requantize32>(cpu_has_sme, cpu_has_sme2,
                             is_supported<sme2_s8q_planar_5x5_s1_4rows_dot_za>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sme2_s8q_planar_5x5_s1_4rows_dot_za(args.cpu_info);
      return new DepthwisePlanar<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_s8q_planar_5x5_s2_4rows_dot_za",
    constraint<Requantize32>(cpu_has_sme, cpu_has_sme2,
                             is_supported<sme2_s8q_planar_5x5_s2_4rows_dot_za>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sme2_s8q_planar_5x5_s2_4rows_dot_za(args.cpu_info);
      return new DepthwisePlanar<int8_t>(strat, args, qp);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             qp_weights_are_symmetric,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>,
                             qp_has_no_left_shift,
                             has_channel_multiplier,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<int8_t, int8_t, int8_t, int32_t, false>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>,
                             qp_has_no_left_shift,
                             has_channel_multiplier,
                             cpu_has_sve2),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new sve_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<int8_t, int8_t, int8_t, int32_t, false>(strat, args, qp);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             has_no_channel_multiplier,
                             qp_weights_are_symmetric,
                             qp_has_no_left_shift,
                             cpu_has_dot_product),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift,
                             cpu_has_dot_product),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_3x3_s1_output2x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst>,
                             has_no_channel_multiplier,
                             qp_has_no_left_shift),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_nhwc_generic_output3x3_mla_depthfirst",
    constraint<Requantize32>(has_no_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto kernel = new a64_s8q_nhwc_generic_output9_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstStrategy<int8_t>(kernel, 3, 3, args);
      return new DepthwiseDepthfirstGeneric<int8_t>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst>,
                             qp_has_no_left_shift,
                             has_channel_multiplier,
                             cpu_has_dot_product),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<int8_t, int8_t, int8_t, int32_t, false>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst",
    constraint<Requantize32>(is_supported<a64_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst>,
                             qp_has_no_left_shift,
                             has_channel_multiplier,
                             cpu_has_dot_product),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto strat = new a64_s8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<int8_t, int8_t, int8_t, int32_t, false>(strat, args, qp);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_s8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    constraint<Requantize32>(has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Requantize32 &qp) -> DepthwiseCommon<int8_t, int8_t, int8_t> * {
      auto kern = new a64_s8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstMultiplierStrategy<int8_t>(kern, args);
      return new DepthwiseDepthfirstMultiplier<int8_t, int8_t, int8_t, int32_t, true>(strat, args, qp);
    },
  },
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<int8_t, int8_t, int8_t, Requantize32> *depthwise_implementation_list()
{
  return depthwise_s8q_methods;
}

template UniqueDepthwiseCommon<int8_t, int8_t, int8_t> depthwise(const DepthwiseArgs &, const Requantize32 &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, int8_t, Requantize32>(const DepthwiseArgs &, const Requantize32 &);

}  // namespace depthwise
}  // namespace arm_conv
