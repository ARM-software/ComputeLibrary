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

#pragma once

#include "arm_compute/core/Error.h"
#include "depthfirst_driver.hpp"
#include "utils.hpp"
#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

namespace arm_conv {
namespace pooling {

template <typename TInput, typename TOutput, typename OutputStage = Nothing>
class IGenericDepthfirstStrategy;

template <typename TInput, typename TOutput>
class IGenericDepthfirstStrategy<TInput, TOutput, Nothing>
{
  public:
  virtual ~IGenericDepthfirstStrategy() = default;

  typedef void (*KernelType)(
    uint64_t window_cells,
    uint64_t n_valid_cells,
    uint64_t n_channels,
    const TInput *const *,
    TOutput *
  );

  virtual KernelType get_kernel(void) const = 0;
};

template <typename TInput, typename TOutput>
class IGenericDepthfirstStrategy<TInput, TOutput, Requantize32>
{
  public:
  virtual ~IGenericDepthfirstStrategy() = default;

  typedef void (*KernelType)(
    uint64_t window_cells,
    uint64_t n_valid_cells,
    uint64_t n_channels,
    const TInput *const *,
    TOutput *,
    const Requantize32 &
  );

  virtual KernelType get_kernel(void) const = 0;
};

template <typename TInput, typename TOutput, typename OutputStage>
struct Invoker;

template <typename TInput, typename TOutput>
struct Invoker<TInput, TOutput, Nothing>
{
  static inline void invoke(
    const typename IGenericDepthfirstStrategy<TInput, TOutput, Nothing>::KernelType kern,
    uint64_t window_cells,
    uint64_t n_valid_cells,
    uint64_t n_channels,
    const TInput *const *inptrs,
    TOutput *outptr,
    const Nothing &
  )
  {
    kern(window_cells, n_valid_cells, n_channels, inptrs, outptr);
  }
};

template <typename TInput, typename TOutput>
struct Invoker<TInput, TOutput, Requantize32>
{
  static inline void invoke(
    const typename IGenericDepthfirstStrategy<TInput, TOutput, Requantize32>::KernelType kern,
    uint64_t window_cells,
    uint64_t n_valid_cells,
    uint64_t n_channels,
    const TInput *const *inptrs,
    TOutput *outptr,
    const Requantize32 &qp
  )
  {
    kern(window_cells, n_valid_cells, n_channels, inptrs, outptr, qp);
  }
};

template <typename TInput, typename TOutput, typename OutputStage>
class GenericDepthfirstWrapper : public IDepthfirstStrategy
{
  using StratType = IGenericDepthfirstStrategy<TInput, TOutput, OutputStage>;

  std::unique_ptr<const StratType> m_strat;
  const unsigned int window_rows, window_cols;

  public:
  GenericDepthfirstWrapper(const StratType *strat, const PoolingArgs &args)
  : m_strat(strat), window_rows(args.pool_window.rows), window_cols(args.pool_window.cols)
  {
  }

  unsigned int get_input_rows(void) const override { return window_rows; }
  unsigned int get_input_cols(void) const override { return window_cols; }
  unsigned int get_output_rows(void) const override { return 1; }
  unsigned int get_output_cols(void) const override { return 1; }

  typename StratType::KernelType get_kernel(void) const { return m_strat->get_kernel(); }
};

template <typename TInput, typename TOutput=TInput, typename OutputStage=Nothing>
class PoolingDepthfirstGeneric : public DepthfirstDriver<TInput, TOutput>
{
  const OutputStage m_os;

  protected:
  size_t get_working_size_per_thread(unsigned int) const override { return 0; }
  void initialise_working_space(void *, unsigned int) const override { /* Nothing */ }

  /* Compute a portion of the output tensor with padding. */
  void compute_tile_padded(
    unsigned int output_i, unsigned int output_j,
    unsigned int channel_start, unsigned int channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *
  ) const override
  {
    // Determine start position and padding
    const int start_i = static_cast<int>(output_i * this->m_args.pool_stride.rows) - this->m_args.padding.top;
    const auto input_i = static_cast<unsigned int>(start_i < 0 ? 0 : start_i);
    const auto pad_top = static_cast<unsigned int>(start_i < 0 ? -start_i : 0);
    const int end_i = start_i + this->m_args.pool_window.rows;
    const auto pad_bottom = static_cast<unsigned int>((unsigned int) end_i < this->m_args.input_rows ? 0 : end_i - this->m_args.input_rows);
    const auto valid_rows = this->m_args.pool_window.rows - (pad_top + pad_bottom);

    const int start_j = static_cast<int>(output_j * this->m_args.pool_stride.cols) - this->m_args.padding.left;
    const auto input_j = static_cast<unsigned int>(start_j < 0 ? 0 : start_j);
    const auto pad_left = static_cast<unsigned int>(start_j < 0 ? -start_j : 0);
    const int end_j = start_j + this->m_args.pool_window.cols;
    const auto pad_right = static_cast<unsigned int>((unsigned int) end_j < this->m_args.input_cols ? 0 : end_j - this->m_args.input_cols);
    const auto valid_cols = this->m_args.pool_window.cols - (pad_left + pad_right);

    // Determine the number of valid cells and prepare the pointers
    const auto n_valid_cells = valid_rows * valid_cols;
    auto inptrs = reinterpret_cast<const TInput **>(alloca(n_valid_cells * sizeof(TInput *)));
    {
      auto my_ptr = inptrs;
      auto row_ptr = input.base + input_i*input.ld_row + input_j*input.ld_col + channel_start;
      for (auto i = valid_rows; i; i--)
      {
        auto ptr = row_ptr;
        row_ptr += input.ld_row;

        for (auto j = valid_cols; j; j--)
        {
          *(my_ptr++) = ptr;
          ptr += input.ld_col;
        }
      }
    }

    auto outptr = output.base + output_i*output.ld_row + output_j*output.ld_col + channel_start;

    // Some padding variants include (or exclude) the padding values; we handle
    // this by computing the extent of the padded input tensor and hence
    // computing the total number of cells captured in the pooling window.
    const auto bottom_padded_height = this->m_args.input_rows + this->m_args.padding.bottom;
    const auto captured_rows = std::min<int>(end_i, bottom_padded_height) - start_i;
    const auto right_padded_width = this->m_args.input_cols + this->m_args.padding.right;
    const auto captured_cols = std::min<int>(end_j, right_padded_width) - start_j;
    const auto captured_cells = captured_rows * captured_cols;
    const auto window_cells = this->m_args.exclude_padding ? n_valid_cells : captured_cells;

    // Execute the kernel
    Invoker<TInput, TOutput, OutputStage>::invoke(
      reinterpret_cast<const GenericDepthfirstWrapper<TInput, TOutput, OutputStage> *>(this->m_strat.get())->get_kernel(),
      window_cells, n_valid_cells, channel_end - channel_start, inptrs, outptr, m_os
    );
  }

  // Compute a portion of the work with only top/bottom padding.
  void compute_row_padded_tile_row(
    const unsigned int output_i, unsigned int output_j, unsigned int n_tile_cols,
    const unsigned int channel_start, const unsigned int channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *working_space
  ) const override
  {
    ARM_COMPUTE_UNUSED(working_space);
    // Determine start position and padding
    const int start_i = static_cast<int>(output_i * this->m_args.pool_stride.rows) - this->m_args.padding.top;
    const auto input_i = static_cast<unsigned int>(start_i < 0 ? 0 : start_i);
    const auto pad_top = static_cast<unsigned int>(start_i < 0 ? -start_i : 0);
    const int end_i = start_i + this->m_args.pool_window.rows;
    const auto pad_bottom = static_cast<unsigned int>((unsigned int) end_i < this->m_args.input_rows ? 0 : end_i - this->m_args.input_rows);
    const auto valid_rows = this->m_args.pool_window.rows - (pad_top + pad_bottom);

    const int start_j = static_cast<int>(output_j * this->m_args.pool_stride.cols) - this->m_args.padding.left;
    const auto input_j = static_cast<unsigned int>(start_j < 0 ? 0 : start_j);
    const auto valid_cols = this->m_args.pool_window.cols;

    // Determine the number of valid cells and prepare the pointers
    const auto n_valid_cells = valid_rows * valid_cols;
    auto inptrs = reinterpret_cast<const TInput **>(alloca(n_valid_cells * sizeof(TInput *)));
    {
      auto my_ptr = inptrs;
      auto row_ptr = input.base + input_i*input.ld_row + input_j*input.ld_col + channel_start;
      for (auto i = valid_rows; i; i--)
      {
        auto ptr = row_ptr;
        row_ptr += input.ld_row;

        for (auto j = valid_cols; j; j--)
        {
          *(my_ptr++) = ptr;
          ptr += input.ld_col;
        }
      }
    }

    auto outptr = output.base + output_i*output.ld_row + output_j*output.ld_col + channel_start;

    // Some padding variants include (or exclude) the padding values; we handle
    // this by computing the extent of the padded input tensor and hence
    // computing the total number of cells captured in the pooling window.
    const auto bottom_padded_height = this->m_args.input_rows + this->m_args.padding.bottom;
    const auto captured_rows = std::min<int>(end_i, bottom_padded_height) - start_i;
    const auto captured_cells = captured_rows * valid_cols;
    const auto window_cells = this->m_args.exclude_padding ? n_valid_cells : captured_cells;

    for (; n_tile_cols; n_tile_cols--)
    {
      // Execute the kernel
      Invoker<TInput, TOutput, OutputStage>::invoke(
        reinterpret_cast<const GenericDepthfirstWrapper<TInput, TOutput, OutputStage> *>(this->m_strat.get())->get_kernel(),
        window_cells, n_valid_cells, channel_end - channel_start, inptrs, outptr, m_os
      );

      // Update the pointers; the output strides by a column and the inputs
      // stride by a number of columns.
      outptr += output.ld_col;
      for (auto n = 0u; n < n_valid_cells; n++)
      {
        inptrs[n] += this->m_args.pool_stride.cols * input.ld_col;
      }
    }
  }

  public:
  PoolingDepthfirstGeneric(
    const IGenericDepthfirstStrategy<TInput, TOutput, OutputStage> *strat,
    const PoolingArgs &args,
    const OutputStage &os = {}
  )
  : DepthfirstDriver<TInput, TOutput>(
      new GenericDepthfirstWrapper<TInput, TOutput, OutputStage>(strat, args),
      args
    ),
    m_os(os)
  {
  }
};

}  // namespace pooling
}  // namespace arm_conv
