/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#include "pooling.hpp"

#include <cstddef>
#include <functional>
#include <cstring>

namespace arm_conv {
namespace pooling {

template <typename TInput, typename TOutput, class OutputStage = Nothing>
struct PoolingImplementation
{
  const PoolingMethod method;
  const char * name;
  std::function<bool(const PoolingArgs &, const OutputStage &)> is_supported;
  std::function<uint64_t(const PoolingArgs &, const OutputStage &)> cycle_estimate;
  std::function<PoolingCommon<TInput, TOutput> *(const PoolingArgs &, const OutputStage &)> initialise;

  bool get_is_supported(const PoolingArgs &args, const OutputStage &os) const
  {
    return (is_supported == nullptr) ? true : is_supported(args, os);
  }

  uint64_t get_cycle_estimate(const PoolingArgs &args, const OutputStage &os) const
  {
    return (cycle_estimate == nullptr) ? 0 : cycle_estimate(args, os);
  }

  PoolingCommon<TInput, TOutput> *get_instance(const PoolingArgs &args, const OutputStage &os) const
  {
    return initialise(args, os);
  }
};

/**
 * \relates PoolingImplementation
 */
template <typename TInput, typename TOutput, class OutputStage = Nothing>
const PoolingImplementation<TInput, TOutput, OutputStage> *pooling_implementation_list();

template <typename TInput, typename TOutput, class OutputStage = Nothing>
bool find_implementation(
  const PoolingArgs &args,
  const OutputStage &os,
  const PoolingImplementation<TInput, TOutput, OutputStage> * &selected
)
{
  // For now, return the first valid implementation
  const auto *impl = pooling_implementation_list<TInput, TOutput, OutputStage>();
  for (; impl->method != PoolingMethod::DEFAULT; impl++)
  {
    if (args.config != nullptr)
    {
      // Apply filters provided by the configuration
      const auto cfg = args.config;

      if (cfg->filter != "" && !std::strstr(impl->name, cfg->filter.c_str()))
      {
        continue;
      }
    }

    if (impl->get_is_supported(args, os))
    {
      selected = impl;
      return true;
    }
  }
  return false;
}

template <typename TInput, typename TOutput, class OutputStage>
UniquePoolingCommon<TInput, TOutput> pooling(const PoolingArgs &args, const OutputStage &os)
{
  const PoolingImplementation<TInput, TOutput, OutputStage> *impl = nullptr;
  const bool success = find_implementation<TInput, TOutput, OutputStage>(args, os, impl);
  return UniquePoolingCommon<TInput, TOutput>(success ? impl->get_instance(args, os) : nullptr);
}

template <class Strategy>
bool is_supported(const PoolingArgs &args, const Nothing &)
{
  return ((args.pool_type == Strategy::pooling_type) &&
          (args.pool_window.rows == Strategy::pool_rows) &&
          (args.pool_window.cols == Strategy::pool_cols) &&
          (args.pool_stride.rows == Strategy::stride_rows) &&
          (args.pool_stride.cols == Strategy::stride_cols));
}

}  //  namespace pooling
}  //  namespace arm_conv
