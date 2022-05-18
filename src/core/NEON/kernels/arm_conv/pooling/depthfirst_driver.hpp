/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"

namespace arm_conv {
namespace pooling {

class IDepthfirstStrategy
{
  public:
  virtual ~IDepthfirstStrategy() = default;

  virtual unsigned int get_input_rows() const = 0;
  virtual unsigned int get_input_cols() const = 0;

  virtual unsigned int get_output_rows() const = 0;
  virtual unsigned int get_output_cols() const = 0;
};


template <typename T>
struct TensorSpec
{
  T base;
  size_t ld_row, ld_col;

  TensorSpec(T ptr, size_t ld_row, size_t ld_col)
  : base(ptr), ld_row(ld_row), ld_col(ld_col) {}
};


template <typename TInput, typename TOutput>
class DepthfirstDriver : public PoolingCommon<TInput, TOutput>
{
  protected:
  using Parent = PoolingCommon<TInput, TOutput>;

  // The strategy which we're applying to solve the pooling problem.
  std::unique_ptr<const IDepthfirstStrategy> m_strat;

  /* Compute the amount of working space required for a single thread. */
  virtual size_t get_working_size_per_thread(unsigned int n_input_channels) const = 0;

  /* Initialise the working space for a thread. */
  virtual void initialise_working_space(void *, unsigned int n_input_channels) const = 0;

  /* Compute a portion of the output tensor with padding. */
  virtual void compute_tile_padded(
    unsigned int output_i, unsigned int output_j,
    unsigned int output_channel_start, unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *working_space
  ) const = 0;

  /* Compute a portion of the work with only top/bottom padding.
   *
   * The default implementation of this repeatedly calls into the padded tile
   * variant.
   */
  virtual void compute_row_padded_tile_row(
    const unsigned int output_i, unsigned int output_j, unsigned int n_tile_cols,
    const unsigned int output_channel_start, const unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *working_space
  ) const
  {
    for (; n_tile_cols; n_tile_cols--, output_j += m_strat->get_output_cols())
    {
      this->compute_tile_padded(
        output_i, output_j, output_channel_start, output_channel_end,
        input, output, working_space
      );
    }
  }

  /* Compute a portion of the output tensor with no padding.
   *
   * The default implementation of this repeatedly calls into the padded
   * variant.
   */
  virtual void compute_tiles_unpadded(
    unsigned int start_output_i, unsigned int start_output_j,
    unsigned int n_tile_rows, unsigned int n_tile_cols,
    unsigned int output_channel_start, unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    void *working_space
  ) const
  {
    for (unsigned int tile_i = 0; tile_i < n_tile_rows; tile_i++)
    {
      this->compute_row_padded_tile_row(
        start_output_i, start_output_j, n_tile_cols,
        output_channel_start, output_channel_end,
        input, output, working_space
      );
      start_output_i += m_strat->get_output_rows();
    }
  }

  void execute_internal(
    unsigned int n_batches,
    unsigned int input_height,
    unsigned int input_width,
    unsigned int n_channels,
    const PaddingValues &padding,
    const void *input,
    size_t ld_input_col,
    size_t ld_input_row,
    size_t ld_input_batch,
    unsigned int output_height,
    unsigned int output_width,
    void *output,
    size_t ld_output_col,
    size_t ld_output_row,
    size_t ld_output_batch,
    void *working_space,
    unsigned int thread_id,
    unsigned int n_threads
  ) const override
  {
    // Get and initialise the working space for this thread.
    void *thread_working_space =
      static_cast<uint8_t *>(working_space) + thread_id * this->get_working_size_per_thread(n_channels);
    this->initialise_working_space(thread_working_space, n_channels);

    // Construct convenient representations of the input/output tensors.
    TensorSpec<const TInput *> input_tensor(reinterpret_cast<const TInput *>(input), ld_input_row, ld_input_col);
    TensorSpec<TOutput *> output_tensor(reinterpret_cast<TOutput *>(output), ld_output_row, ld_output_col);

    // If the output is a 1x1 tensor, which commonly occurs at the end of a
    // network, then we change the threading strategy to parallelise over
    // channels rather than rows of the tensor.
    if (n_threads > 1 && output_height == 1 && output_width == 1)
    {
      // Determine how many channels should be assigned to each thread, we
      // round up first to ensure we get a reasonable spread across the
      // threads.
      const auto channels_per_thread = arm_gemm::roundup(arm_gemm::roundup(n_channels, 16u), n_threads) / n_threads;
      const auto start_channel = thread_id * channels_per_thread;
      const auto end_channel = std::min(start_channel + channels_per_thread, n_channels);

      if (start_channel >= end_channel)
      {
        // This thread should move on if we have insufficient work to do.
        return;
      }

      for (; n_batches; n_batches--)
      {
        // We know we don't need to iterate over rows or columns here; so just
        // execute the tile.
        this->compute_tile_padded(
          0, 0,  // Compute the only output point
          start_channel, end_channel,
          input_tensor, output_tensor, thread_working_space
        );

        // Progress the pointers for the next batch.
        input_tensor.base += ld_input_batch;
        output_tensor.base += ld_output_batch;
      }

      // Exit here, since we've done all the work using the different strategy.
      return;
    }

    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
      // Iterate over rows of the output tensor; we stripe over the tiles.
      for (unsigned int start_output_i = thread_id * m_strat->get_output_rows();
           start_output_i < output_height;
           start_output_i += n_threads * m_strat->get_output_rows())
      {
        // Determine what (if any padding) is required on the top/bottom of
        // this row of the convolution.
        const auto end_output_i = start_output_i + m_strat->get_output_rows();
        const bool pad_output_bottom = output_height < end_output_i;

        const int start_input_i = start_output_i * this->m_args.pool_stride.rows - padding.top;
        const bool pad_input_top = start_input_i < 0;
        const int end_input_i = start_input_i + m_strat->get_input_rows();
        const bool pad_input_bottom = static_cast<int>(input_height) < end_input_i;
        const bool pad_row = pad_input_top || pad_input_bottom || pad_output_bottom;

        // Iterate over the columns of the output tensor; we attempt to grab as
        // much as possible of the unpadded regions, so the loop structure is a
        // bit odd.
        unsigned int start_output_j = 0;
        while (start_output_j < output_width)
        {
          const int start_in_j = start_output_j * this->m_args.pool_stride.cols - padding.left;
          const bool pad_input_left = start_in_j < 0;

          // Determine if we can process a number of unpadded tiles in one go.
          int n_unpadded_tiles = 0;
          if (!pad_input_left)
          {
            // Determine the maximum number of tiles we could handle.
            n_unpadded_tiles = (output_width - start_output_j) / m_strat->get_output_cols();

            // Handle padding on the right hand edge
            const int tile_stride = m_strat->get_output_cols() * this->m_args.pool_stride.cols;
            int end_output_j = start_output_j + n_unpadded_tiles * m_strat->get_output_cols();
            int end_input_j = start_in_j + m_strat->get_input_cols() + (n_unpadded_tiles - 1)*tile_stride;

            while (n_unpadded_tiles > 0 &&
                   (static_cast<int>(output_width) < end_output_j ||
                    static_cast<int>(input_width) < end_input_j))
            {
              n_unpadded_tiles--;
              end_output_j -= m_strat->get_output_cols();
              end_input_j -= tile_stride;
            }
          }

          // Process unpadded tiles, if possible, otherwise process a padded tile.
          if (n_unpadded_tiles)
          {
            if (!pad_row)
            {
              // Completely unpadded execution
              this->compute_tiles_unpadded(
                start_output_i, start_output_j,
                1, n_unpadded_tiles,  // Compute a row of unpadded tiles
                0, n_channels,  // Compute all channels
                input_tensor, output_tensor, thread_working_space
              );
            }
            else
            {
              // Top/bottom padding only
              this->compute_row_padded_tile_row(
                start_output_i, start_output_j, n_unpadded_tiles,
                0, n_channels,  // Compute all channels
                input_tensor, output_tensor, thread_working_space
              );
            }
            start_output_j += n_unpadded_tiles * m_strat->get_output_cols();
          }
          else
          {
            this->compute_tile_padded(
              start_output_i, start_output_j,
              0, n_channels,  // Compute all channels
              input_tensor, output_tensor, thread_working_space
            );
            start_output_j += m_strat->get_output_cols();
          }
        }
      }

      // Progress the pointers for the next batch.
      input_tensor.base += ld_input_batch;
      output_tensor.base += ld_output_batch;
    }
  }

  public:
  DepthfirstDriver(const IDepthfirstStrategy *strategy, const PoolingArgs &args)
  : Parent(args), m_strat(strategy)
  {
  }

  size_t get_working_size(unsigned int n_threads) const override
  {
    return this->get_working_size(n_threads, this->m_args.n_channels);
  }

  size_t get_working_size(unsigned int n_threads, unsigned int n_channels) const override final
  {
    return n_threads * this->get_working_size_per_thread(n_channels);
  }
};

}  // namespace pooling
}  // namespace arm_conv
