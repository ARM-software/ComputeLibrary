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

#pragma once

#include "depthwise_depthfirst_generic.hpp"

#include "arm_gemm.hpp"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

using arm_gemm::Requantize32;

namespace arm_conv {
namespace depthwise {

template <class Strategy, unsigned OutputRows, unsigned int OutputCols>
class DepthwiseDepthfirstGenericQuantized : public DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>
{
  using Parent = DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>;
  using TInput = typename Parent::TInput;
  using TAccum = typename Parent::TAccum;
  using TOutput = typename Parent::TOutput;

  Requantize32 m_qp;

  public:
  DepthwiseDepthfirstGenericQuantized(const DepthwiseArgs &args, const Requantize32 &qp)
    : Parent(args), m_qp(qp)
  {
  }

  DepthwiseDepthfirstGenericQuantized(DepthwiseDepthfirstGenericQuantized &) = delete;
  DepthwiseDepthfirstGenericQuantized &operator=(DepthwiseDepthfirstGenericQuantized &) = delete;

  void pack_parameters(void *buffer, const void *biases, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    m_qp.bias = static_cast<const TAccum *>(biases);
    Parent::pack_parameters(buffer, biases, weights, ld_weight_col, ld_weight_row);
  }

  using DepthwiseDepthfirstGenericBase<Strategy, OutputRows, OutputCols>::execute;
  void execute(
    const unsigned int batches,
    const unsigned int input_height,
    const unsigned int input_width,
    const unsigned int input_channels,
    const PaddingValues &padding,
    const void *const _input,
    const size_t ld_input_col,
    const size_t ld_input_row,
    const size_t ld_input_batch,
    const void *const parameters,
    const unsigned int output_height,
    const unsigned int output_width,
    void *const _output,
    const size_t ld_output_col,
    const size_t ld_output_row,
    const size_t ld_output_batch,
    void *const _working_space,
    const unsigned int thread_id,
    const unsigned int n_threads
  ) const override
  {
    Strategy strat(this->m_args.cpu_info);
#ifdef CYCLE_PROFILING
    arm_gemm::profiler prof;
#endif

    // Create a function to initialise the input buffer
    const auto initialise_input_buffer = [this] (TInput *const buffer, const unsigned int n) {
      std::memset(buffer, static_cast<TInput>(m_qp.a_offset), n * sizeof(TInput));
    };

    // Create a function to execute a tile of work
    const auto tile_fn = [&] (const TInput *const *const inptrs, TOutput *const * const outptrs) {
#ifdef CYCLE_PROFILING
      auto p = prof.ScopedProfiler(
        PROFILE_KERNEL,
        (unsigned long) (OutputRows * OutputCols * this->m_args.kernel_rows* this->m_args.kernel_cols)
      );
#endif
      strat.kernel(inptrs, outptrs, parameters, m_qp,
                   this->m_args.kernel_rows * this->m_args.kernel_cols,
                   this->m_args.input_channels);
    };

    // Call into a parent utility function to do the actual work.
    Parent::execute_tiles(
      tile_fn, initialise_input_buffer,
      batches, input_height, input_width, input_channels, padding,
      _input, ld_input_col, ld_input_row, ld_input_batch,
      output_height, output_width,
      _output, ld_output_col, ld_output_row, ld_output_batch,
      _working_space, thread_id, n_threads
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv
