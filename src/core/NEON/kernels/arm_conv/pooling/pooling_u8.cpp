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

#include "pooling_implementation.hpp"
#include "pooling_depthfirst.hpp"
#include "pooling_depthfirst_generic.hpp"

#include "kernels/cpp_nhwc_1x1_stride_any_depthfirst.hpp"
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SME)
#include "kernels/sme_u8_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/sme_u8_nhwc_max_2x2_s1_output2x2_depthfirst.hpp"
#include "kernels/sme_u8_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SME)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/sve_u8_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst.hpp"
#include "kernels/sve_u8_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst.hpp"
#include "kernels/a64_u8_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/a64_u8_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(__aarch64__)

#include <cstdint>

namespace arm_conv {
namespace pooling {

static const PoolingImplementation<uint8_t, uint8_t> pooling_u8_methods[] = {
  {
    PoolingMethod::DEPTHFIRST,
    "cpp_u8_nhwc_1x1_stride_any_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      return args.pool_window.rows == 1 && args.pool_window.cols == 1;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new cpp_nhwc_1x1_stride_any_depthfirst<uint8_t>(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SME)
  {
    PoolingMethod::DEPTHFIRST,
    "sme_u8_nhwc_max_2x2_s1_output2x2_depthfirst",
    [] (const PoolingArgs &args, const Nothing &os) -> bool {
      return args.cpu_info->has_sme() &&
             is_supported<sme_u8_nhwc_max_2x2_s1_output2x2_depthfirst>(args, os);
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sme_u8_nhwc_max_2x2_s1_output2x2_depthfirst(args.cpu_info);
      return new PoolingDepthfirst<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sme_u8_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      // This kernel can only be used when there is either no padding, or we don't care
      // about the value of the padding. Otherwise, we would need to pass in the zero-point
      // for the quantization regime.
      return (args.exclude_padding ||
              (args.padding.top == 0 && args.padding.bottom == 0 &&
               args.padding.left == 0 && args.padding.right == 0)
              ) && args.pool_type == PoolingType::AVERAGE &&
             args.cpu_info->has_sme2();
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sme_u8_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sme_u8_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      return args.cpu_info->has_sme() && args.pool_type == PoolingType::MAX;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sme_u8_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SME)
#if defined(ARM_COMPUTE_ENABLE_SVE)
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst",
    [] (const PoolingArgs &args, const Nothing &os) -> bool {
      return args.cpu_info->has_sve() &&
             is_supported<sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst>(args, os);
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst(args.cpu_info);
      return new PoolingDepthfirst<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      // This kernel can only be used when there is either no padding, or we don't care
      // about the value of the padding. Otherwise, we would need to pass in the zero-point
      // for the quantization regime.
      return (args.exclude_padding ||
              (args.padding.top == 0 && args.padding.bottom == 0 &&
               args.padding.left == 0 && args.padding.right == 0)
              ) && args.pool_type == PoolingType::AVERAGE &&
             args.cpu_info->has_sve2();
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sve_u8_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      return args.cpu_info->has_sve() && args.pool_type == PoolingType::MAX;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new sve_u8_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  {
    PoolingMethod::DEPTHFIRST,
    "a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst",
    is_supported<a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst>,
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst(args.cpu_info);
      return new PoolingDepthfirst<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "a64_u8_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      // This kernel can only be used when there is either no padding, or we don't care
      // about the value of the padding. Otherwise, we would need to pass in the zero-point
      // for the quantization regime.
      return (args.exclude_padding ||
              (args.padding.top == 0 && args.padding.bottom == 0 &&
               args.padding.left == 0 && args.padding.right == 0)
              ) && args.pool_type == PoolingType::AVERAGE;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new a64_u8_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "a64_u8_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool { return args.pool_type == PoolingType::MAX; },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      auto strat = new a64_u8_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<uint8_t>(strat, args);
    },
  },
#endif  // defined(__aarch64__)
  { PoolingMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const PoolingImplementation<uint8_t, uint8_t> *pooling_implementation_list()
{
  return pooling_u8_methods;
}

template UniquePoolingCommon<uint8_t, uint8_t> pooling(const PoolingArgs &, const Nothing &);

}  //  namespace pooling
}  //  namespace arm_conv
