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

#include "depthfirst_driver.hpp"
#include "src/core/NEON/kernels/arm_conv/addressing.hpp"
#include "utils.hpp"
#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */
#include <limits>

namespace arm_conv {
namespace pooling {

template <typename TInput, typename TOutput>
class DepthfirstStrategy : public IDepthfirstStrategy
{
  unsigned int input_rows, input_cols, output_rows, output_cols;

  public:
  DepthfirstStrategy(unsigned int window_rows, unsigned int window_cols,
                     unsigned int stride_rows, unsigned int stride_cols,
                     unsigned int output_rows, unsigned int output_cols)
  : input_rows(output_rows + (window_rows - 1) * stride_rows),
    input_cols(output_cols + (window_cols - 1) * stride_cols),
    output_rows(output_rows), output_cols(output_cols)
  {
  }

  unsigned int get_input_rows() const override { return input_rows; }
  unsigned int get_input_cols() const override { return input_cols; }
  unsigned int get_output_rows() const override { return output_rows; }
  unsigned int get_output_cols() const override { return output_cols; }

  typedef void (*KernelType)(
    unsigned int n_channels,
    const TInput *const *,
    TOutput *const *,
    bool exclude_padding,
    unsigned int pad_left,
    unsigned int pad_top,
    unsigned int pad_right,
    unsigned int pad_bottom
  );
  virtual KernelType get_kernel(void) const = 0;
};


struct WorkingSpace
{
  void *input_buffer;
  void *output_buffer;
};


template <typename TInput, typename TOutput=TInput, class OutputStage=Nothing>
class PoolingDepthfirst : public DepthfirstDriver<TInput, TOutput>
{
  size_t sizeof_input_buffer(void) const
  {
    return sizeof(TInput) * this->m_args.n_channels;
  }

  size_t sizeof_output_buffer(void) const
  {
    return sizeof(TOutput) * this->m_args.n_channels;
  }

  protected:
  /* Compute the amount of working space required for a single thread. */
  size_t get_working_size_per_thread(unsigned int n_channels) const override
  {
    return sizeof(WorkingSpace) + n_channels * (sizeof(TInput) + sizeof(TOutput));
  }

  /* Initialise the working space for a thread. */
  void initialise_working_space(void *raw_ws, unsigned int n_channels) const override
  {
    auto ws = reinterpret_cast<WorkingSpace *>(raw_ws);
    ws->input_buffer = ws + 1;
    ws->output_buffer = reinterpret_cast<TInput *>(ws + 1) + n_channels;

    // Fill the input buffer with an appropriate value
    TInput fill_val = 0;
    if (this->m_args.pool_type == PoolingType::MAX)
    {
      using limits = std::numeric_limits<TInput>;
      if (limits::has_infinity)
      {
        fill_val = -limits::infinity();
      }
      else
      {
        fill_val = limits::min();
      }
    }

    auto ptr = reinterpret_cast<TInput *>(ws->input_buffer);
    for (; n_channels; n_channels--)
    {
      *(ptr++) = fill_val;
    }
  }

  /* Compute a portion of the output tensor with padding. */
  void compute_tile_padded(
    unsigned int output_i, unsigned int output_j,
    unsigned int channel_start, unsigned int channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *working_space
  ) const override
  {
    const auto kern = reinterpret_cast<const DepthfirstStrategy<TInput, TOutput> *>(
      this->m_strat.get())->get_kernel();

    // Get the working space, and some space on the stack for pointer arrays
    auto ws = reinterpret_cast<WorkingSpace *>(working_space);
    auto inptr_array = reinterpret_cast<const TInput **>(alloca(
        sizeof(TInput *) * this->m_strat->get_input_rows() * this->m_strat->get_input_cols()));
    auto outptr_array = reinterpret_cast<TOutput **>(alloca(
        sizeof(TOutput *) * this->m_strat->get_output_rows() * this->m_strat->get_output_cols()));

    // Prepare the input pointers
    const int ii = static_cast<int>(output_i * this->m_args.pool_stride.rows) - this->m_args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);
    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);

    const unsigned int end_ii = ii + this->m_strat->get_input_rows();
    const auto input_pad_bottom = end_ii < this->m_args.input_rows ? 0 : end_ii - this->m_args.input_rows;

    const int ij = static_cast<int>(output_j * this->m_args.pool_stride.cols) - this->m_args.padding.left;
    const auto input_pad_left = static_cast<unsigned int>(ij < 0 ? -ij : 0);
    const auto input_j = static_cast<unsigned int>(ij < 0 ? 0 : ij);

    const unsigned int end_ij = ij + this->m_strat->get_input_cols();
    const auto input_pad_right = end_ij < this->m_args.input_cols ? 0 : end_ij - this->m_args.input_cols;

    fill_pointer_array<const TInput>(
      inptr_array, this->m_strat->get_input_rows(), this->m_strat->get_input_cols(),
      input.base + input_i*input.ld_row + input_j*input.ld_col + channel_start,
      input.ld_row, input.ld_col,
      reinterpret_cast<const TInput *>(ws->input_buffer),
      input_pad_top, this->m_args.input_rows - input_i,
      input_pad_left, this->m_args.input_cols - input_j
    );

    // Prepare the output pointers
    fill_pointer_array(
      outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + channel_start,
      output.ld_row, output.ld_col,
      reinterpret_cast<TOutput *>(ws->output_buffer),
      0, this->m_args.output_rows - output_i, // Top padding, # valid rows
      0, this->m_args.output_cols - output_j  // Left padding, # valid columns
    );

    // Call the kernel
    kern(
      channel_end - channel_start, inptr_array, outptr_array,
      this->m_args.exclude_padding,
      input_pad_left, input_pad_top,
      input_pad_right, input_pad_bottom
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
    const auto kern = reinterpret_cast<const DepthfirstStrategy<TInput, TOutput> *>(
      this->m_strat.get())->get_kernel();

    // Get the working space, and some space on the stack for pointer arrays
    auto ws = reinterpret_cast<WorkingSpace *>(working_space);
    auto inptr_array = reinterpret_cast<const TInput **>(alloca(
        sizeof(TInput *) * this->m_strat->get_input_rows() * this->m_strat->get_input_cols()));
    auto outptr_array = reinterpret_cast<TOutput **>(alloca(
        sizeof(TOutput *) * this->m_strat->get_output_rows() * this->m_strat->get_output_cols()));

    // Prepare the initial input pointers
    const int ii = static_cast<int>(output_i * this->m_args.pool_stride.rows) - this->m_args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);
    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);

    const unsigned int end_ii = ii + this->m_strat->get_input_rows();
    const auto input_pad_bottom = end_ii < this->m_args.input_rows ? 0 : end_ii - this->m_args.input_rows;

    const int ij = static_cast<int>(output_j * this->m_args.pool_stride.cols) - this->m_args.padding.left;
    const auto input_j = static_cast<unsigned int>(ij < 0 ? 0 : ij);

    const auto end_oi = output_i + this->m_strat->get_output_cols();
    const auto output_pad_bottom = end_oi < this->m_args.output_rows ? 0 : end_oi - this->m_args.output_rows;

    fill_pointer_array<const TInput>(
      inptr_array, this->m_strat->get_input_rows(), this->m_strat->get_input_cols(),
      input.base + input_i*input.ld_row + input_j*input.ld_col + channel_start,
      input.ld_row, input.ld_col,
      reinterpret_cast<const TInput *>(ws->input_buffer),
      input_pad_top, this->m_args.input_rows - input_i,
      0, this->m_args.input_cols - input_j
    );

    // Prepare the initial output pointers
    fill_pointer_array(
      outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + channel_start,
      output.ld_row, output.ld_col,
      reinterpret_cast<TOutput *>(ws->output_buffer),
      0, this->m_args.output_rows - output_i, // Top padding, # valid rows
      0, this->m_args.output_cols - output_j  // Left padding, # valid columns
    );

    // Call the kernel
    for (; n_tile_cols; n_tile_cols--)
    {
      kern(
        channel_end - channel_start, inptr_array, outptr_array,
        this->m_args.exclude_padding,
        0, input_pad_top,
        0, input_pad_bottom
      );

      // Progress the input and output pointer arrays
      const auto input_col_stride = input.ld_col * this->m_strat->get_output_cols() * this->m_args.pool_stride.cols;
      for (
        auto n = input_pad_top * this->m_strat->get_input_cols();
        n < (this->m_strat->get_input_rows() - input_pad_bottom) * this->m_strat->get_input_cols();
        n++
      )
      {
        inptr_array[n] += input_col_stride;
      }

      const auto output_col_stride = output.ld_col * this->m_strat->get_output_cols();
      for (
        auto n = 0u;
        n < (this->m_strat->get_output_rows() - output_pad_bottom) * this->m_strat->get_output_cols();
        n++
      )
      {
        outptr_array[n] += output_col_stride;
      }
    }
  }

  public:
  PoolingDepthfirst(const DepthfirstStrategy<TInput, TOutput> *strat,
                    const PoolingArgs &args, const OutputStage &os = {})
  : DepthfirstDriver<TInput, TOutput>(strat, args)
  {
    ARM_COMPUTE_UNUSED(os);
  }
};

}  // namespace pooling
}  // namespace arm_conv
