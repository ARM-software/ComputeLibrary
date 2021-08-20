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
#include "depthwise_depthfirst.hpp"
#include "depthwise_depthfirst_generic.hpp"
#include "depthwise_depthfirst_multiplier.hpp"
#include "depthwise_depthfirst_generic_multiplier.hpp"

#include "depthwise_implementation_constraints.hpp"

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
#endif  // defined(__aarch64__)

namespace arm_conv {
namespace depthwise {

namespace
{
  template <class Strategy>
  unsigned int cycle_estimate(const DepthwiseArgs &args, const Nothing &)
  {
    // First-pass: compute the number of output pixels which will be computed.
    return arm_gemm::roundup(args.output_rows, Strategy::output_rows) *
           arm_gemm::roundup(args.output_cols, Strategy::output_cols) *
           arm_gemm::iceildiv(
            (long unsigned) args.input_channels * args.channel_multiplier,
            arm_gemm::utils::get_vector_length<typename Strategy::return_type>(Strategy::vl_type)
          );
  }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  unsigned int not_preferred(const DepthwiseArgs &, const Nothing &)
  {
    return std::numeric_limits<unsigned int>::max();
  }

  unsigned int not_preferred_if_no_multiplier(const DepthwiseArgs &args, const Nothing &)
  {
    return args.channel_multiplier > 1 ? 0 : std::numeric_limits<unsigned int>::max();
  }
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
}

#if defined(__ARM_FP16_ARGS)

static const DepthwiseImplementation<__fp16, __fp16> depthwise_fp16_methods[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
              has_no_channel_multiplier,
              cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>(args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirst<a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_generic_output3x3_mla_depthfirst",
    constraint(has_no_channel_multiplier),
    not_preferred,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirstGeneric<a64_fp16_nhwc_generic_output9_mla_depthfirst, 3, 3>(args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    nullptr,
    not_preferred_if_no_multiplier,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      return new DepthwiseDepthfirstGenericWithMultiplier<a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst>(args);
    },
  },
#endif  // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<__fp16> *depthwise_implementation_list()
{
  return depthwise_fp16_methods;
}

template UniqueDepthwiseCommon<__fp16> depthwise(const DepthwiseArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16>(const DepthwiseArgs &, const Nothing &);

#endif  // defined(__ARM_FP16_ARGS)

}  // namespace depthwise
}  // namespace arm_conv
