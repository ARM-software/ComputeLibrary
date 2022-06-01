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

/* Utilities for constructing functions which constrain which kernels are
 * selected for a given depthwise problem.
 *
 * It is expected that this will be included in the files which list the
 * available kernels. To avoid multiple definitions, an anonymous namespace is
 * used.
 */

#pragma once

#include "arm_gemm.hpp"
#include "src/core/NEON/kernels/assembly/depthwise.hpp"

namespace arm_conv
{
namespace depthwise
{
namespace
{

template <class OutputStage>
using ConstraintFn = std::function<bool(const DepthwiseArgs &, const OutputStage &)>;

using GenericConstraintFn = std::function<bool(const DepthwiseArgs &, const void *)>;

GenericConstraintFn make_constraint(const GenericConstraintFn &f) __attribute__ ((unused));
GenericConstraintFn make_constraint(const GenericConstraintFn &f)
{
  return f;
}

template <typename ... Fs>
GenericConstraintFn make_constraint(const GenericConstraintFn &f, Fs ... fs)
{
  return [f, fs...] (const DepthwiseArgs &args, const void *os) -> bool {
    return f(args, os) && make_constraint(fs...)(args, os);
  };
}

template <typename OutputStage=Nothing, typename ... Fs>
ConstraintFn<OutputStage> constraint(Fs ... fs)
{
  return [fs...] (const DepthwiseArgs &args, const OutputStage &os) -> bool {
    return make_constraint(fs...)(args, &os);
  };
}

// Some useful constraints
template <class Strategy>
bool is_supported(const DepthwiseArgs &args, const void *)
{
  return ((args.kernel_rows == Strategy::kernel_rows) &&
          (args.kernel_cols == Strategy::kernel_cols) &&
          (args.stride_rows == Strategy::stride_rows) &&
          (args.stride_cols == Strategy::stride_cols));
}

bool cpu_has_dot_product(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_dot_product(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_dotprod();
}

bool cpu_has_sme(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_sme(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_sme();
}

bool cpu_has_sme2(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_sme2(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_sme2();
}

bool cpu_has_sve(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_sve(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_sve();
}

bool cpu_has_sve2(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_sve2(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_sve2();
}

bool cpu_has_fp16(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool cpu_has_fp16(const DepthwiseArgs &args, const void *)
{
  return args.cpu_info->has_fp16();
}

bool has_no_channel_multiplier(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool has_no_channel_multiplier(const DepthwiseArgs &args, const void *)
{
  return args.channel_multiplier == 1;
}

bool has_channel_multiplier(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool has_channel_multiplier(const DepthwiseArgs &args, const void *)
{
  return args.channel_multiplier > 1;
}

// Planar kernels require a "priming" step before the main processing loop.  The kernels can prime with left padding
// or input data, but not right padding - which could be needed in some extreme cases such as a 5x5 kernel, width 1
// padding 2.  These are rare enough and can be handled with other kernels anyway, so filter them out with this.
bool no_prime_right_pad(const DepthwiseArgs &args, const void *) __attribute__ ((unused));
bool no_prime_right_pad(const DepthwiseArgs &args, const void *)
{
  return (args.input_cols + args.padding.left) >= (args.kernel_cols - 1);
}

bool qp_has_no_left_shift(const DepthwiseArgs &args, const void *_qp) __attribute__ ((unused));
bool qp_has_no_left_shift(const DepthwiseArgs &, const void *_qp)
{
  const auto qp = static_cast<const arm_gemm::Requantize32 *>(_qp);
  return qp->per_channel_requant ?
    (qp->per_channel_left_shifts == nullptr) :
    (qp->per_layer_left_shift == 0);
}

bool qp_zero_a_offset(const DepthwiseArgs &args, const void *_qp) __attribute__ ((unused));
bool qp_zero_a_offset(const DepthwiseArgs &, const void *_qp)
{
  const auto qp = static_cast<const arm_gemm::Requantize32 *>(_qp);
  return qp->a_offset == 0;
}

template <typename T> bool qp_skip_clamp(const DepthwiseArgs &args, const void *_qp) __attribute__ ((unused));
template <typename T> bool qp_skip_clamp(const DepthwiseArgs &, const void *_qp)
{
  const auto qp = static_cast<const arm_gemm::Requantize32 *>(_qp);
  return (qp->minval == std::numeric_limits<T>::min() &&
          qp->maxval == std::numeric_limits<T>::max());
}

}  // namespace
}  // namespace depthwise
}  // namespace arm_conv
