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
#include "kernels/sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp32_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst.hpp"
#include "kernels/sve_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst.hpp"
#include "kernels/sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp32_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/a64_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst.hpp"
#include "kernels/a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst.hpp"
#include "kernels/a64_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
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

#if defined(__aarch64__)
  unsigned int not_preferred(const DepthwiseArgs &, const Nothing &)
  {
    return std::numeric_limits<unsigned int>::max();
  }

  bool fast_mode_enabled(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
  bool fast_mode_enabled(const DepthwiseArgs &args, const void *)
  {
    return args.fast_mode;
  }
#endif // defined(__aarch64__)
}

static const DepthwiseImplementation<float, float> depthwise_fp32_methods[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst>,
              has_no_channel_multiplier,
              cpu_has_sve),
    cycle_estimate<sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               has_no_channel_multiplier,
               cpu_has_sve),
    cycle_estimate<sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_generic_output3x3_mla_depthfirst",
    constraint(has_no_channel_multiplier, cpu_has_sve),
    not_preferred,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto kern = new sve_fp32_nhwc_generic_output9_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstStrategy<float>(kern, 3, 3, args);
      return new DepthwiseDepthfirstGeneric<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst",
    constraint(is_supported<sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst>,
               cpu_has_sve, has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst",
    constraint(is_supported<sve_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst>,
               cpu_has_sve, has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new sve_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp32_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    constraint(cpu_has_sve, has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto kern = new sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstMultiplierStrategy<float>(kern, args);
      return new DepthwiseDepthfirstMultiplier<float, float, float, float, true>(strat, args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_nhwc_3x3_s1_output4x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst>,
                            has_no_channel_multiplier),
    cycle_estimate<a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               has_no_channel_multiplier),
    cycle_estimate<a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_generic_output3x3_mla_depthfirst",
    constraint(has_no_channel_multiplier),
    not_preferred,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto kern = new a64_fp32_nhwc_generic_output9_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstStrategy<float>(kern, 3, 3, args);
      return new DepthwiseDepthfirstGeneric<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst",
    constraint(is_supported<a64_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst>,
               has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst",
    constraint(is_supported<a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst>,
               has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto strat = new a64_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirstMultiplier<float>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp32_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    constraint(has_channel_multiplier),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<float, float, float> * {
      auto kern = new a64_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstMultiplierStrategy<float>(kern, args);
      return new DepthwiseDepthfirstMultiplier<float, float, float, float, true>(strat, args);
    },
  },
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<float> *depthwise_implementation_list()
{
  return depthwise_fp32_methods;
}

template UniqueDepthwiseCommon<float> depthwise(const DepthwiseArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<float>(const DepthwiseArgs &, const Nothing &);

}  // namespace depthwise
}  // namespace arm_conv
