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

#include "src/core/NEON/kernels/arm_conv/addressing.hpp"
#include "depthwise_strategies_common.hpp"
#include "working_space.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

#include <limits>

namespace arm_conv {
namespace depthwise {

template <typename TInput, typename TWeight, typename TOutput, typename TAccum,
          typename OutputStage>
class DepthwiseDepthfirstStrategyCommon
  : public DepthfirstStrategy<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  protected:
  unsigned int m_output_rows, m_output_cols;
  unsigned int m_kernel_rows, m_kernel_cols;
  unsigned int m_stride_rows, m_stride_cols;

  public:
  DepthwiseDepthfirstStrategyCommon(
    unsigned int output_rows, unsigned int output_cols,
    unsigned int kernel_rows, unsigned int kernel_cols,
    unsigned int stride_rows=1, unsigned int stride_cols=1
  ) : m_output_rows(output_rows), m_output_cols(output_cols),
      m_kernel_rows(kernel_rows), m_kernel_cols(kernel_cols),
      m_stride_rows(stride_rows), m_stride_cols(stride_cols)
  {
  }

  DepthwiseDepthfirstStrategyCommon(unsigned int output_size, unsigned int kernel_size, unsigned int stride=1)
  : DepthwiseDepthfirstStrategyCommon(output_size, output_size, kernel_size, kernel_size, stride, stride)
  {
  }

  virtual ~DepthwiseDepthfirstStrategyCommon() {}

  unsigned int get_output_rows() const override { return m_output_rows; }
  unsigned int get_output_cols() const override { return m_output_cols; }

  unsigned int get_kernel_rows() const override { return m_kernel_rows; }
  unsigned int get_kernel_cols() const override { return m_kernel_cols; }

  unsigned int get_stride_rows() const override { return m_stride_rows; }
  unsigned int get_stride_cols() const override { return m_stride_cols; }
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class DepthwiseDepthfirstStrategy : public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  using Parent = DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>;

  public:
  using Parent::Parent;

  typedef void (*IndirectKernelType)(
    const TInput *const *input_ptrs,
    TOutput *const *output_ptrs,
    const void *params,
    unsigned int n_channels,
    const TAccum activation_min,
    const TAccum activation_max
  );
  virtual IndirectKernelType get_indirect_kernel(void) const = 0;

  typedef void (*DirectKernelType)(
    const unsigned int n_tile_rows, const unsigned int n_tile_cols,
    const TInput *inptr_base, int64_t ld_input_row, int64_t ld_input_col,
    TOutput *outptr_base, int64_t ld_output_row, int64_t ld_output_col,
    const void *params, unsigned int n_channels,
    const TAccum activation_min,
    const TAccum activation_max
  );
  virtual DirectKernelType get_direct_kernel(void) const = 0;
};

template <typename TInput, typename TWeight, typename TOutput>
class DepthwiseDepthfirstStrategy<TInput, TWeight, TOutput, int32_t>
: public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>
{
  using Parent = DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>;

  protected:
  interleaves::PackingArguments get_packing_args(void) const
  {
    return interleaves::PackingArguments(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      false, sizeof(int32_t),  // Don't pack the bias
      this->get_vl_type(), sizeof(int32_t), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
  }

  public:
  using Parent::Parent;

  typedef void (*KernelType)(
    unsigned int,  //  n_channels,
    const TInput *const *,  // inptrs
    const TWeight *,  // weights
    const int32_t *,  //  bias,
    const arm_gemm::Requantize32 &,
    const int32_t *, const int32_t *,  //  requant_muls and requant_shifts
    TOutput *const *  // outptrs
  );
  virtual KernelType get_kernel() const = 0;

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleaves::get_storage_size_generic(get_packing_args(), args);
  }

  void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const arm_gemm::Requantize32 &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const override
  {
    interleaves::pack_parameters_generic(
      get_packing_args(), args, buffer, biases, weights, ld_weight_col, ld_weight_row);
  }
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
class DepthwiseDepthfirstCommon : public DepthfirstDriver<TInput, TWeight, TOutput>
{
  using StratType = DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>;
  OutputStage m_os;

  protected:
  inline OutputStage &get_output_stage(void) { return m_os; }
  inline const OutputStage &get_output_stage(void) const { return m_os; }

  public:
  DepthwiseDepthfirstCommon(StratType *const strat, const DepthwiseArgs &args, const OutputStage &os)
  : DepthfirstDriver<TInput, TWeight, TOutput>(strat, args), m_os(os)
  {
  }

  DepthwiseDepthfirstCommon(DepthwiseDepthfirstCommon &) = delete;
  DepthwiseDepthfirstCommon &operator=(DepthwiseDepthfirstCommon &) = delete;

  size_t get_storage_size(void) const override
  {
    return reinterpret_cast<const StratType *>(this->m_strat.get())->
      get_storage_size(this->m_args);
  }

  void pack_parameters(void *buffer, const void *biases, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    reinterpret_cast<const StratType *>(this->m_strat.get())->
      pack_parameters(this->m_args, buffer, biases, m_os, weights, ld_weight_col, ld_weight_row);
  }
};

namespace depthwise_depthfirst {

/* Workspace Element for an array of input pointers as consumed by the
 * specialised depthwise kernels.
 */
template <typename T>
class InputArrayElement
{
  public:
  struct Workspace
  {
    const T **inptr_array;
  };

  template <class OutputStage>
  static size_t get_element_size(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    return sizeof(T **) * args.strategy->get_input_rows() * args.strategy->get_input_cols();
  }

  template <class WorkspaceType, class OutputStage>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    ws->inptr_array = reinterpret_cast<const T**>(buffer);
    return reinterpret_cast<char *>(buffer) + get_element_size(args);
  }
};

template <typename TAccum, typename OutputStage, bool IsDot=false>
struct WorkspaceFinalElement
{
  using Element = ActivationsElement<TAccum, OutputStage>;
};

template <>
struct WorkspaceFinalElement<int32_t, arm_gemm::Requantize32, false>
{
  using Element = RequantizationParametersElement;
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
struct Invoke
{
  constexpr static bool supports_direct_kernel = true;

  template <typename Strat, typename Workspace>
  static inline void indirect(const Strat *strat, const Workspace *ws, const OutputStage &, const void *params, const TAccum *, unsigned int n_channels)
  {
    strat->get_indirect_kernel()(
      ws->inptr_array,
      ws->outptr_array,
      params, n_channels,
      ws->activation_min, ws->activation_max
    );
  }

  template <typename Strat, typename Workspace>
  static void direct(
    const Strat *strat, const Workspace *ws, const OutputStage &,
    unsigned int n_tile_rows, unsigned int n_tile_cols,
    const TInput *inptr, size_t ld_in_row, size_t ld_in_col,
    TOutput *outptr, size_t ld_out_row, size_t ld_out_col,
    const void *params, unsigned int n_channels
  )
  {
    strat->get_direct_kernel()(
      n_tile_rows, n_tile_cols,
      inptr, ld_in_row, ld_in_col,
      outptr, ld_out_row, ld_out_col,
      params, n_channels, ws->activation_min, ws->activation_max
    );
  }
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum>
struct Invoke<TInput, TWeight, TOutput, TAccum, arm_gemm::Requantize32>
{
  constexpr static bool supports_direct_kernel = false;

  template <typename Strat, typename Workspace>
  static inline void indirect(const Strat *strat, const Workspace *ws, const arm_gemm::Requantize32 &qp, const void *params, const TAccum *, unsigned int n_channels)
  {
    strat->get_kernel()(
      n_channels, ws->inptr_array,
      reinterpret_cast<const TWeight *>(params), ws->bias,
      qp, ws->requant_muls, ws->requant_shifts,
      ws->outptr_array
    );
  }

  template <typename Strat, typename Workspace>
  static inline void direct(
    const Strat *, const Workspace *, const arm_gemm::Requantize32 &,
    unsigned int, unsigned int,  // n_tile_rows, n_tile_cols
    const TInput *, size_t, size_t,  // Input pointer, row stride, column stride
    TOutput *, size_t, size_t,  // Output pointer, row stride, column stride
    const void *, unsigned int  // Parameters, number of channels
  )
  {
    // Do nothing - this should never be reached because entry to it is guarded
    // by an `if` on a `constexpr static bool`.
  }
};

namespace
{

template <typename OutputStage>
inline void stash_bias(OutputStage &, const void *) {}

template <>
inline void stash_bias(arm_gemm::Requantize32 &qp, const void *bias) __attribute__ ((unused));

template <>
inline void stash_bias(arm_gemm::Requantize32 &qp, const void *bias)
{
  qp.bias = reinterpret_cast<const int32_t *>(bias);
}

}

}  // namespace depthwise_depthfirst

template <typename TInput,
          typename TWeight=TInput,
          typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TInput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class DepthwiseDepthfirst
: public DepthwiseDepthfirstCommon<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  using StratType = DepthwiseDepthfirstStrategy<TInput, TWeight, TOutput, TAccum>;
  using Parent = DepthwiseDepthfirstCommon<TInput, TWeight, TOutput, TAccum, OutputStage>;
  using WorkspaceManager = Workspace<
    OutputArrayElement<TOutput>,
    depthwise_depthfirst::InputArrayElement<TInput>,
    InputBufferElement<TInput>,
    typename depthwise_depthfirst::WorkspaceFinalElement<TAccum, OutputStage>::Element
  >;
  using WorkingSpace = typename WorkspaceManager::WorkspaceType;

  // We keep a copy of the bias and output stage
  const TAccum *m_bias;

  public:
  DepthwiseDepthfirst(StratType *const strat, const DepthwiseArgs &args, const OutputStage &os = {})
  : Parent(strat, args, os), m_bias(nullptr)
  {
  }

  DepthwiseDepthfirst(DepthwiseDepthfirst &) = delete;
  DepthwiseDepthfirst &operator=(DepthwiseDepthfirst &) = delete;

  void pack_parameters(void *buffer, const void *biases, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    reinterpret_cast<const StratType *>(this->m_strat.get())->pack_parameters(
      this->m_args, buffer, biases, this->get_output_stage(),
      weights, ld_weight_col, ld_weight_row
    );
    m_bias = reinterpret_cast<const TAccum *>(biases);
    depthwise_depthfirst::stash_bias(this->get_output_stage(), biases);
  }

  size_t get_working_size_per_thread(const unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    return WorkspaceManager::get_sizeof_workspace(
      WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, this->get_output_stage())
    );
  }

  void initialise_working_space(void *buffer, unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    WorkspaceManager::initialise(
      buffer, WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, this->get_output_stage())
    );
  }

  protected:
  void compute_tile_padded(
    const DepthwiseArgs &args,
    unsigned int output_i, unsigned int output_j,
    unsigned int output_channel_start, unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    const void *parameters,
    void *working_space_raw
  ) const override
  {
    // Get the working space
    auto ws = reinterpret_cast<WorkingSpace *>(working_space_raw);

    // Compute the input pointer array
    const auto input_channel_start = output_channel_start / args.channel_multiplier;

    const int ii = static_cast<int>(output_i * args.stride_rows) - args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);
    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);

    const int ij = static_cast<int>(output_j * args.stride_cols) - args.padding.left;
    const auto input_pad_left = static_cast<unsigned int>(ij < 0 ? -ij : 0);
    const auto input_j = static_cast<unsigned int>(ij < 0 ? 0 : ij);

    fill_pointer_array<const TInput>(
      ws->inptr_array, this->m_strat->get_input_rows(), this->m_strat->get_input_cols(),
      input.base + input_i*input.ld_row + input_j*input.ld_col + input_channel_start,
      input.ld_row, input.ld_col,
      ws->input_buffer,
      input_pad_top, args.input_rows - input_i,
      input_pad_left, args.input_cols - input_j
    );

    // Compute the output pointer array
    fill_pointer_array(
      ws->outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + output_channel_start,
      output.ld_row, output.ld_col,
      ws->output_buffer,
      0, args.output_rows - output_i, // Top padding, # valid rows
      0, args.output_cols - output_j  // Left padding, # valid columns
    );

    // Execute the kernel
    depthwise_depthfirst::Invoke<TInput, TWeight, TOutput, TAccum, OutputStage>::indirect(
      reinterpret_cast<const StratType *>(this->m_strat.get()),
      ws, this->get_output_stage(), parameters, m_bias, output_channel_end - output_channel_start
    );
  }

  void compute_row_padded_tile_row(
    const DepthwiseArgs &args,
    const unsigned int output_i, unsigned int output_j, unsigned int n_tile_cols,
    const unsigned int output_channel_start, const unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    const void *parameters,
    void *working_space
  ) const override
  {
    using Invoker = depthwise_depthfirst::Invoke<TInput, TWeight, TOutput, TAccum, OutputStage>;
    auto ws = reinterpret_cast<WorkingSpace *>(working_space);
    const auto strat = reinterpret_cast<const StratType *>(this->m_strat.get());
    const auto os = this->get_output_stage();

    // Compute top and bottom padding; hence fill in the initial pointer arrays.
    const auto input_channel_start = output_channel_start / args.channel_multiplier;
    const int ii = static_cast<int>(output_i * args.stride_rows) - args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);

    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);
    const auto input_j = output_j * args.stride_cols - args.padding.left;

    // Valid input rows is the smallest of the input rows that aren't padding for this tile, and the number of rows
    // available.
    const auto valid_input_rows = std::min(strat->get_input_rows() - input_pad_top, args.input_rows - input_i);
    const auto valid_output_rows = std::min(strat->get_output_rows(), args.output_rows - output_i);

    const auto input_point_stride = input.ld_col * this->m_strat->get_output_cols() * args.stride_cols;
    const auto output_point_stride = output.ld_col * this->m_strat->get_output_cols();

    fill_pointer_array<const TInput>(
      ws->inptr_array, this->m_strat->get_input_rows(), this->m_strat->get_input_cols(),
      input.base + input_i*input.ld_row + input_j*input.ld_col + input_channel_start,
      input.ld_row, input.ld_col,
      ws->input_buffer,
      input_pad_top, args.input_rows - input_i,
      0, args.input_cols - input_j  // No left padding
    );

    fill_pointer_array(
      ws->outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + output_channel_start,
      output.ld_row, output.ld_col,
      ws->output_buffer,
      0, args.output_rows - output_i,  // Top padding, # valid rows
      0, args.output_cols - output_j  // Left padding, # valid columns
    );

    for (; n_tile_cols; n_tile_cols--)
    {
      // Execute the kernel
      Invoker::indirect(
        strat, ws, os, parameters, m_bias, output_channel_end - output_channel_start
      );

      // Update all unpadded pointers
      {
        auto ptr = ws->inptr_array + strat->get_input_cols() * input_pad_top;
        for (auto n = input_pad_top; n < (valid_input_rows + input_pad_top); n++)
        {
          for (auto m = 0u; m < strat->get_input_cols(); m++)
          {
            *(ptr++) += input_point_stride;
          }
        }
      }
      {
        auto ptr = ws->outptr_array;
        for (auto n = 0u; n < valid_output_rows * strat->get_output_cols(); n++)
        {
          *(ptr++) += output_point_stride;
        }
      }
    }
  }

  void compute_tiles_unpadded(
    const DepthwiseArgs &args,
    unsigned int output_i, const unsigned int output_j,
    unsigned int n_tile_rows, unsigned int n_tile_cols,
    unsigned int output_channel_start, unsigned int output_channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    const void *parameters,
    void *working_space_raw
  ) const override
  {
    using Invoker = depthwise_depthfirst::Invoke<TInput, TWeight, TOutput, TAccum, OutputStage>;
    auto ws = reinterpret_cast<WorkingSpace *>(working_space_raw);
    const auto strat = reinterpret_cast<const StratType *>(this->m_strat.get());
    const auto os = this->get_output_stage();

    if (Invoker::supports_direct_kernel)
    {
      // If the direct kernel is supported, then use it.
      // Compute the base pointers we'll use in the tile.
      auto outptr = output.base + output_channel_start + output_i * output.ld_row + output_j * output.ld_col;
      const int start_input_i = output_i * args.stride_rows - args.padding.top;
      const int start_input_j = output_j * args.stride_cols - args.padding.left;
      auto inptr = input.base + output_channel_start + start_input_i * input.ld_row + start_input_j * input.ld_col;

      // Execute the kernel
      Invoker::direct(
        strat, ws, os,
        n_tile_rows, n_tile_cols,
        inptr, input.ld_row, input.ld_col,
        outptr, output.ld_row, output.ld_col,
        parameters, output_channel_end - output_channel_start
      );
    }
    else
    {
      // Otherwise, we repeatedly call the padded kernel but use our knowledge
      // of the tensor structure to avoid recomputing the pointer array.
      const auto input_channel_start = output_channel_start / args.channel_multiplier;

      const auto n_input_pointers = this->m_strat->get_input_rows() * this->m_strat->get_input_cols();
      const auto input_point_stride = input.ld_col * this->m_strat->get_output_cols() * args.stride_cols;
      const auto n_output_pointers = this->m_strat->get_output_rows() * this->m_strat->get_output_cols();
      const auto output_point_stride = output.ld_col * this->m_strat->get_output_cols();

      // For each tile row, initialise the input and output pointer arrays. For
      // each subsequent tile we simply update the pointers.
      for (unsigned int tile_i = 0; tile_i < n_tile_rows; tile_i++)
      {
        const int input_i = static_cast<int>(output_i * args.stride_rows) - args.padding.top;
        const int input_j = static_cast<int>(output_j * args.stride_cols) - args.padding.left;

        fill_pointer_array<const TInput>(
          ws->inptr_array, this->m_strat->get_input_rows(), this->m_strat->get_input_cols(),
          input.base + input_i*input.ld_row + input_j*input.ld_col + input_channel_start,
          input.ld_row, input.ld_col,
          ws->input_buffer,
          0, args.input_rows,
          0, args.input_cols
        );

        // Compute the output pointer array
        fill_pointer_array(
          ws->outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
          output.base + output_i*output.ld_row + output_j*output.ld_col + output_channel_start,
          output.ld_row, output.ld_col,
          ws->output_buffer,
          0, args.output_rows,
          0, args.output_cols
        );

        for (unsigned int tile_j = 0; tile_j < n_tile_cols; tile_j++)
        {
          // Invoke the indirect kernel for this tile
          depthwise_depthfirst::Invoke<TInput, TWeight, TOutput, TAccum, OutputStage>::indirect(
            strat, ws, os, parameters, m_bias, output_channel_end - output_channel_start
          );

          // Progress the pointers
          for (auto i = 0u; i < n_input_pointers; i++)
          {
            ws->inptr_array[i] += input_point_stride;
          }
          for (auto i = 0u; i < n_output_pointers; i++)
          {
            ws->outptr_array[i] += output_point_stride;
          }
        }

        output_i += this->m_strat->get_output_rows();
      }
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
