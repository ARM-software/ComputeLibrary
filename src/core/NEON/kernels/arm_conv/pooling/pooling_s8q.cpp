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
#include "pooling_depthfirst_generic.hpp"

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SME)
#include "kernels/sme_s8q_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/sme_s8q_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SME)
#if defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/sve_s8q_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/sve_s8q_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
#include "kernels/a64_s8q_nhwc_avg_generic_depthfirst.hpp"
#include "kernels/a64_s8q_nhwc_max_generic_depthfirst.hpp"
#endif  // defined(__aarch64__)

#include <cstdint>

namespace arm_conv {
namespace pooling {

static const PoolingImplementation<int8_t, int8_t, Requantize32> pooling_s8q_methods[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SME)
  {
    PoolingMethod::DEPTHFIRST,
    "sme_s8q_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool {
      return args.cpu_info->has_sme2() && args.pool_type == PoolingType::AVERAGE;
    },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new sme_s8q_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sme_s8q_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool {
      return args.cpu_info->has_sme2() && args.pool_type == PoolingType::MAX;
    },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new sme_s8q_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SME)
#if defined(ARM_COMPUTE_ENABLE_SVE)
  {
    PoolingMethod::DEPTHFIRST,
    "sve_s8q_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool {
      return args.cpu_info->has_sve2() && args.pool_type == PoolingType::AVERAGE;
    },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "sve_s8q_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool {
      return args.cpu_info->has_sve2() && args.pool_type == PoolingType::MAX;
    },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new sve_s8q_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  {
    PoolingMethod::DEPTHFIRST,
    "a64_s8q_nhwc_avg_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool {
      return args.pool_type == PoolingType::AVERAGE;
    },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_avg_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
  {
    PoolingMethod::DEPTHFIRST,
    "a64_s8q_nhwc_max_generic_depthfirst",
    [] (const PoolingArgs &args, const Requantize32 &) -> bool { return args.pool_type == PoolingType::MAX; },
    nullptr,
    [] (const PoolingArgs &args, const Requantize32 &rq) -> PoolingCommon<int8_t, int8_t> * {
      auto strat = new a64_s8q_nhwc_max_generic_depthfirst(args.cpu_info);
      return new PoolingDepthfirstGeneric<int8_t, int8_t, Requantize32>(strat, args, rq);
    },
  },
#endif  // defined(__aarch64__)
  { PoolingMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const PoolingImplementation<int8_t, int8_t, Requantize32> *pooling_implementation_list()
{
  return pooling_s8q_methods;
}

template UniquePoolingCommon<int8_t, int8_t> pooling(const PoolingArgs &, const Requantize32 &);

}  //  namespace pooling
}  //  namespace arm_conv
