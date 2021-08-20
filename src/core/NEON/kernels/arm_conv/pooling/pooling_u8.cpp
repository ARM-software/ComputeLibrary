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

#include "pooling_implementation.hpp"
#include "pooling_depthfirst.hpp"
#include "pooling_depthfirst_generic.hpp"

#include "kernels/cpp_nhwc_1x1_stride_any_depthfirst.hpp"
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(ARM_COMPUTE_ENABLE_SVE2)
#include "kernels/sve_u8_nhwc_avg_generic_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE2)
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

namespace
{
  template <class Strategy>
  bool is_supported(const PoolingArgs &args, const Nothing &)
  {
    return ((args.pool_type == Strategy::pooling_type()) &&
            (args.pool_window.rows == Strategy::pool_rows()) &&
            (args.pool_window.cols == Strategy::pool_cols()) &&
            (args.pool_stride.rows == Strategy::stride_rows()) &&
            (args.pool_stride.cols == Strategy::stride_cols()));
  }
}

static const PoolingImplementation<uint8_t, uint8_t> pooling_u8_methods[] = {
  {
    PoolingMethod::DEPTHFIRST,
    "cpp_u8_nhwc_1x1_stride_any_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      return args.pool_window.rows == 1 && args.pool_window.cols == 1;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirstGeneric<cpp_nhwc_1x1_stride_any_depthfirst<uint8_t>>(args);
    },
  },
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(ARM_COMPUTE_ENABLE_SVE2)
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool {
      // This kernel can only be used when there is either no padding, or we don't care
      // about the value of the padding. Otherwise, we would need to pass in the zero-point
      // for the quantization regime.
      return args.cpu_info->has_sve2() && (args.exclude_padding ||
              (args.padding.top == 0 && args.padding.bottom == 0 &&
               args.padding.left == 0 && args.padding.right == 0)
              ) && args.pool_type == PoolingType::AVERAGE;
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirstGeneric<sve_u8_nhwc_avg_generic_depthfirst>(args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE2)
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst",
    [] (const PoolingArgs &args, const Nothing &unused) -> bool {
      return args.cpu_info->has_sve() && is_supported<sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst>(args, unused);
    },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirst<sve_u8_nhwc_max_2x2_s1_output2x2_depthfirst>(args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sve_u8_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool { return args.cpu_info->has_sve() && args.pool_type == PoolingType::MAX; },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirstGeneric<sve_u8_nhwc_max_generic_depthfirst>(args);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  {
    PoolingMethod::DEPTHFIRST,
    "a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst",
    is_supported<a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst>,
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirst<a64_u8_nhwc_max_2x2_s1_output2x2_depthfirst>(args);
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
      return new PoolingDepthfirstGeneric<a64_u8_nhwc_avg_generic_depthfirst>(args);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "a64_u8_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Nothing &) -> bool { return args.pool_type == PoolingType::MAX; },
    nullptr,
    [] (const PoolingArgs &args, const Nothing &) -> PoolingCommon<uint8_t, uint8_t> * {
      return new PoolingDepthfirstGeneric<a64_u8_nhwc_max_generic_depthfirst>(args);
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
