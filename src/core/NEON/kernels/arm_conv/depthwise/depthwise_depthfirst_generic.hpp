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

#include "depthwise_depthfirst.hpp"

namespace arm_conv {
namespace depthwise {

template <typename TInput, typename TOutput, typename TAccum>
struct GenericDepthfirstKernelStrategyFunctionType
{
  using KernelType = std::function<void(const TInput *const *const, TOutput *const *const, const void *, const void *, const unsigned int, const unsigned int, const TAccum, const TAccum)>;
};

template <typename TInput, typename TOutput>
struct GenericDepthfirstKernelStrategyFunctionType<TInput, TOutput, int32_t>
{
  using KernelType = std::function<void(const TInput *const *const, TOutput *const *const, const void *, const arm_gemm::Requantize32 &, unsigned int, unsigned int)>;
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum>
class GenericDepthfirstKernelStrategy
{
  unsigned int m_n_output_points;
  arm_gemm::VLType m_vl_type;
  unsigned int m_accumulator_depth_vl;

  public:
  GenericDepthfirstKernelStrategy(unsigned int n_output_points, arm_gemm::VLType vl_type, unsigned int accumulator_depth_vl=1)
  : m_n_output_points(n_output_points), m_vl_type(vl_type), m_accumulator_depth_vl(accumulator_depth_vl)
  {
  }

  virtual ~GenericDepthfirstKernelStrategy() = default;

  virtual arm_gemm::VLType get_vl_type() const { return m_vl_type; }
  virtual unsigned int get_accumulator_depth_vl() const { return m_accumulator_depth_vl; }
  virtual unsigned int get_n_output_points() const { return m_n_output_points; }

  using KernelType = typename GenericDepthfirstKernelStrategyFunctionType<TInput, TOutput, TAccum>::KernelType;
  virtual KernelType get_kernel(void) const = 0;
};

template <typename TInput,
          typename TWeight=TInput,
          typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TInput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class GenericDepthfirstStrategy : public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  protected:
  using KernelStrategyType = GenericDepthfirstKernelStrategy<TInput, TWeight, TOutput, TAccum>;
  std::unique_ptr<KernelStrategyType> m_strategy;

  public:
  GenericDepthfirstStrategy(
    KernelStrategyType *strat, unsigned int n_output_rows, unsigned int n_output_cols,
    const DepthwiseArgs &args
  )
  : DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>(
      n_output_rows, n_output_cols,
      args.kernel_rows, args.kernel_cols,
      args.stride_rows, args.stride_cols
    ),
    m_strategy(strat)
  {
  }

  GenericDepthfirstStrategy(GenericDepthfirstStrategy &) = delete;
  GenericDepthfirstStrategy operator=(GenericDepthfirstStrategy &) = delete;

  arm_gemm::VLType get_vl_type(void) const override { return m_strategy->get_vl_type(); }
  unsigned int get_accumulator_depth_vl(void) const override { return m_strategy->get_accumulator_depth_vl(); }

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      false, sizeof(TAccum),  // Don't pack the bias
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    return interleaves::get_storage_size_generic(packing_args, args);
  }

  void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const OutputStage &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const override
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      false, sizeof(TAccum),  // Don't pack the bias
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    interleaves::pack_parameters_generic(
      packing_args, args, buffer, biases, weights, ld_weight_col, ld_weight_row);
  }

  const typename KernelStrategyType::KernelType get_kernel() const { return m_strategy->get_kernel(); }
};

// Use a templated function to marshal arguments when executing the kernel.
template <typename OutputStage> struct DepthwiseDepthfirstGenericKernelCall;

template <>
struct DepthwiseDepthfirstGenericKernelCall<Nothing>
{
  template <typename StratType, typename WorkspaceType, typename TAccum>
  static void execute(
    const StratType *strat, const WorkspaceType *ws, const Nothing &,
    const TAccum *bias, const void *params,
    const unsigned int n_kernel_points, const unsigned int n_output_channels
  )
  {
    strat->get_kernel()(
      ws->inptr_array,
      ws->outptr_array,
      params, bias,
      n_kernel_points, n_output_channels,
      ws->activation_min, ws->activation_max
    );
  }
};

template <>
struct DepthwiseDepthfirstGenericKernelCall<arm_gemm::Requantize32>
{
  template <typename StratType, typename WorkspaceType>
  static void execute(
    const StratType *strat, const WorkspaceType *ws, const arm_gemm::Requantize32 &qp,
    const int32_t *, const void *params,
    const unsigned int n_kernel_points, const unsigned int n_output_channels
  )
  {
    strat->get_kernel()(
      ws->inptr_array,
      ws->outptr_array,
      params, qp,
      n_kernel_points, n_output_channels
    );
  }
};


/* Workspace Element for an array of input pointers as consumed by the
 * "Generic" depthwise kernels.
 */
template <typename T>
class GenericInputArrayElement
{
  public:
  struct Workspace
  {
    const T **inptr_array;
  };

  template <class OutputStage>
  static size_t get_element_size(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    const auto kernel_points = args.depthwise_args.kernel_rows * args.depthwise_args.kernel_cols;
    return sizeof(T **) * args.strategy->get_input_rows() * args.strategy->get_input_cols() * kernel_points;
  }

  template <class WorkspaceType, class OutputStage>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    ws->inptr_array = reinterpret_cast<const T**>(buffer);
    return reinterpret_cast<char *>(buffer) + get_element_size(args);
  }
};

template <typename TInput, typename TWeight=TInput, typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TInput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class DepthwiseDepthfirstGeneric : public DepthwiseDepthfirstCommon<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  using StratType = GenericDepthfirstStrategy<TInput, TWeight, TOutput, TAccum, OutputStage>;
  using Parent = DepthwiseDepthfirstCommon<TInput, TWeight, TOutput, TAccum, OutputStage>;
  using WorkspaceManager = Workspace<
    OutputArrayElement<TOutput>,
    GenericInputArrayElement<TInput>,
    InputBufferElement<TInput>,
    ActivationsElement<TAccum, OutputStage>
  >;
  using WorkingSpace = typename WorkspaceManager::WorkspaceType;
  const TAccum *m_bias = nullptr;

  public:
  DepthwiseDepthfirstGeneric(StratType *const strat, const DepthwiseArgs &args, const OutputStage &os={})
  : Parent(strat, args, os)
  {
  }

  DepthwiseDepthfirstGeneric(DepthwiseDepthfirstGeneric &) = delete;
  DepthwiseDepthfirstGeneric &operator=(DepthwiseDepthfirstGeneric &) = delete;

  void pack_parameters(
    void *buffer, const void *biases,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) override
  {
    Parent::pack_parameters(buffer, biases, weights, ld_weight_col, ld_weight_row);
    m_bias = reinterpret_cast<const TAccum *>(biases);  // Get a copy of the biases
    depthwise_depthfirst::stash_bias(this->get_output_stage(), m_bias);
  }

  size_t get_working_size_per_thread(const unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    return WorkspaceManager::get_sizeof_workspace(WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, this->get_output_stage()));
  }

  void initialise_working_space(void *buffer, unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    return WorkspaceManager::initialise(buffer, WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, this->get_output_stage()));
  }

  protected:
  void compute_tile_padded(
    const DepthwiseArgs &args,
    unsigned int output_i, unsigned int output_j,
    unsigned int channel_start, unsigned int channel_end,
    const TensorSpec<const TInput *> &input,
    const TensorSpec<TOutput *> &output,
    const void *parameters,
    void *working_space_raw
  ) const override
  {
    // Get the working space
    WorkingSpace *ws = reinterpret_cast<WorkingSpace *>(working_space_raw);

    const int ii = static_cast<int>(output_i * args.stride_rows) - args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);
    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);

    const int ij = static_cast<int>(output_j * args.stride_cols) - args.padding.left;
    const auto input_pad_left = static_cast<unsigned int>(ij < 0 ? -ij : 0);
    const auto input_j = static_cast<unsigned int>(ij < 0 ? 0 : ij);

    fill_pointer_array_generic_kernel<const TInput>(
      ws->inptr_array,
      this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      args.kernel_rows, args.kernel_cols,
      args.stride_rows, args.stride_cols,
      input.base + input_i*input.ld_row + input_j*input.ld_col + channel_start,
      input.ld_row, input.ld_col,
      ws->input_buffer,
      input_pad_top, args.input_rows - input_i,
      input_pad_left, args.input_cols - input_j
    );

    // Compute the output pointer array
    fill_pointer_array<TOutput>(
      ws->outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + channel_start,
      output.ld_row, output.ld_col,
      ws->output_buffer,
      0, args.output_rows - output_i, // Top padding, # valid rows
      0, args.output_cols - output_j  // Left padding, # valid columns
    );

    // Execute the kernel
    DepthwiseDepthfirstGenericKernelCall<OutputStage>::execute(
      reinterpret_cast<const StratType *>(this->m_strat.get()), ws,
      this->get_output_stage(), m_bias, parameters,
      args.kernel_rows * args.kernel_cols,
      channel_end - channel_start
    );
  }
};

}  // namespace depthwise
}  // namespace arm_conv
