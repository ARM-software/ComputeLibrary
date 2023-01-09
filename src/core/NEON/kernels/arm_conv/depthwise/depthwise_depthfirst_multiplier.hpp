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
#include "interleaves/generic_quantized_dot_product.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

#include <limits>

namespace arm_conv {
namespace depthwise {

template <typename TInput, typename TWeight, typename TOutput, typename TAccum>
class DepthfirstMultiplierStrategy : public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, Nothing>
{
  using Parent = DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, Nothing>;

  protected:
  virtual interleaves::PackingArguments get_packing_args(const DepthwiseArgs &args) const
  {
    return interleaves::PackingArguments(
      args.kernel_rows, args.kernel_cols, sizeof(TWeight),
      true, sizeof(TAccum),
      this->get_vl_type(),
      sizeof(TAccum), 1,
      [args] (unsigned int pos, unsigned int &x, unsigned int &y) -> bool
      {
        if (pos < args.kernel_rows * args.kernel_cols)
        {
          y = pos % args.kernel_cols;
          x = pos / args.kernel_cols;
          return true;
        }
        return false;
      }
    );
  }

  public:
  using Parent::Parent;

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleaves::get_storage_size_generic(this->get_packing_args(args), args);
  }

  void pack_parameters(const DepthwiseArgs &args, void *buffer, const void *biases, const Nothing &, const void *weights, size_t ld_weight_col, size_t ld_weight_row) const override
  {
    interleaves::pack_parameters_generic(
      this->get_packing_args(args), args,
      buffer, biases, weights, ld_weight_col, ld_weight_row
    );
  }

  using KernelType = std::function<void(
    const TInput *const *,  // Input pointers
    TOutput *const *,  // Output pointers
    const void *,  // Ravelled bias, weights, and quantization parameters
    unsigned int,  // # output channels
    TAccum, TAccum  // Min and max activation clamps
  )>;
  virtual KernelType get_kernel(void) const = 0;
};


template <typename TInput, typename TWeight, typename TOutput>
class DepthfirstMultiplierStrategy<TInput, TWeight, TOutput, int32_t> : public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>
{
  using Parent = DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>;

  public:
  using Parent::Parent;

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleaves::quantized::get_storage_size(args, this->get_vl_type(), this->get_accumulator_depth_vl());
  }

  void pack_parameters(const DepthwiseArgs &args, void *buffer, const void *biases, const arm_gemm::Requantize32 &qp, const void *weights, size_t ld_weight_col, size_t ld_weight_row) const override
  {
    interleaves::quantized::pack_parameters<TWeight>(
      buffer, reinterpret_cast<const int32_t *>(biases),
      reinterpret_cast<const TWeight *>(weights), ld_weight_col, ld_weight_row,
      args, qp, this->get_vl_type(), this->get_accumulator_depth_vl()
    );
  }

  using KernelType = std::function<void(
    const TInput *const *,  // Input pointers
    TOutput *const *,  // Output pointers
    const void *,  // Ravelled bias, weights, and quantization parameters
    unsigned int,  // # output channels
    const arm_gemm::Requantize32 &
  )>;
  virtual KernelType get_kernel(void) const = 0;
};


template <typename TInput, typename TWeight, typename TOutput, typename TAccum>
class GenericDepthfirstMultiplierKernelStrategy
{
  const arm_gemm::VLType m_vl_type;
  const unsigned int m_output_rows, m_output_cols;

  public:
  GenericDepthfirstMultiplierKernelStrategy(unsigned int output_rows, unsigned int output_cols, arm_gemm::VLType vl_type)
  : m_vl_type(vl_type), m_output_rows(output_rows), m_output_cols(output_cols)
  {
  }

  virtual ~GenericDepthfirstMultiplierKernelStrategy() = default;

  arm_gemm::VLType get_vl_type(void) const { return m_vl_type; }
  unsigned int get_output_rows(void) const { return m_output_rows; }
  unsigned int get_output_cols(void) const { return m_output_cols; }

  using KernelType = std::function<void(
    const TInput *const *,  // Input pointers
    TOutput *const *,  // Output pointers
    const TWeight *,  // Ravelled weight parameters
    const TAccum *,  // Bias,
    unsigned int, unsigned int,  // Number of kernel points, number of output channels
    TAccum, TAccum  // Activation minimum and maximum
  )>;
  virtual KernelType get_kernel(void) const = 0;
};

template <typename TInput, typename TWeight, typename TOutput>
class GenericDepthfirstMultiplierKernelStrategy<TInput, TWeight, TOutput, int32_t>
{
  const arm_gemm::VLType m_vl_type;
  const unsigned int m_output_rows, m_output_cols;

  public:
  GenericDepthfirstMultiplierKernelStrategy(unsigned int output_rows, unsigned int output_cols, arm_gemm::VLType vl_type)
  : m_vl_type(vl_type), m_output_rows(output_rows), m_output_cols(output_cols)
  {
  }

  virtual ~GenericDepthfirstMultiplierKernelStrategy() = default;

  arm_gemm::VLType get_vl_type(void) const { return m_vl_type; }
  unsigned int get_output_rows(void) const { return m_output_rows; }
  unsigned int get_output_cols(void) const { return m_output_cols; }

  using KernelType = std::function<void(
    const TInput *const *,  // Input pointers
    TOutput *const *,  // Output pointers
    const TWeight *,  // Ravelled weight parameters
    const int32_t *,  // Bias,
    unsigned int, unsigned int,  // Number of kernel points, number of output channels
    const int32_t *, const int32_t *, const int32_t *,  // Per-channel left-shifts, multipliers, right-shifts (need to account for start channel)
    const arm_gemm::Requantize32 &
  )>;
  virtual KernelType get_kernel(void) const = 0;
};

template <typename TInput,
          typename TWeight=TInput,
          typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TInput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class GenericDepthfirstMultiplierStrategy : public DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>
{
  using KernelStrategyType = GenericDepthfirstMultiplierKernelStrategy<TInput, TWeight, TOutput, TAccum>;
  std::unique_ptr<KernelStrategyType> m_kern;

  protected:
  virtual interleaves::PackingArguments get_packing_args(const DepthwiseArgs &args) const
  {
    return interleaves::PackingArguments(
      args.kernel_rows, args.kernel_cols, sizeof(TWeight),
      false, sizeof(TAccum),
      this->get_vl_type(),
      sizeof(TAccum), 1,
      [args] (unsigned int pos, unsigned int &x, unsigned int &y) -> bool
      {
        if (pos < args.kernel_rows * args.kernel_cols)
        {
          y = pos % args.kernel_cols;
          x = pos / args.kernel_cols;
          return true;
        }
        return false;
      }
    );
  }

  public:
  GenericDepthfirstMultiplierStrategy(KernelStrategyType *kern, const DepthwiseArgs &args)
  : DepthwiseDepthfirstStrategyCommon<TInput, TWeight, TOutput, TAccum, OutputStage>(
      kern->get_output_rows(), kern->get_output_cols(),
      args.kernel_rows, args.kernel_cols,
      args.stride_rows, args.stride_cols
    ),
    m_kern(kern)
  {
  };

  arm_gemm::VLType get_vl_type(void) const override { return m_kern->get_vl_type(); }
  const typename KernelStrategyType::KernelType get_kernel(void) const { return m_kern->get_kernel(); }

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleaves::get_storage_size_generic(this->get_packing_args(args), args);
  }

  void pack_parameters(const DepthwiseArgs &args, void *buffer, const void *biases, const OutputStage &, const void *weights, size_t ld_weight_col, size_t ld_weight_row) const override
  {
    interleaves::pack_parameters_generic(
      this->get_packing_args(args), args,
      buffer, biases, weights, ld_weight_col, ld_weight_row
    );
  }
};

// Specialise elements of the wrapper based on the type of kernel.
namespace depthfirst_multiplier {

/* Working space element which contains a pointer for each row of input, a row
 * of padding, and a space which can be used to construct an NCHW-ordered patch
 * of input.
 */
template <typename T, bool IsGeneric=false, typename OutputStage=Nothing>
class InputPatchElement
{
  public:
  struct Workspace
  {
    constexpr static bool InputPatchIsGeneric = IsGeneric;
    const T **input_rows;
    T *input_padding;
    T *input_patch;
  };

  static size_t get_element_size(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    return sizeof_input_rows(args) + sizeof_input_padding(args) + sizeof_input_patch(args);
  }

  template <class WorkspaceType>
  static void *initialise(WorkspaceType *ws, void *buffer, const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    auto buffer_bytes = reinterpret_cast<char *>(buffer);

    ws->input_rows = reinterpret_cast<const T **>(buffer_bytes);
    buffer_bytes += sizeof_input_rows(args);

    ws->input_padding = reinterpret_cast<T*>(buffer_bytes);
    buffer_bytes += sizeof_input_padding(args);

    ws->input_patch = reinterpret_cast<T*>(buffer_bytes);
    buffer_bytes += sizeof_input_patch(args);

    // Initialise the padding
    memset(ws->input_padding,
           get_input_buffer_fill_value(args.output_stage),
           sizeof_input_padding(args));

    return buffer_bytes;
  }

  protected:
  static size_t sizeof_input_rows(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    if (IsGeneric)
    {
      return sizeof(T *) * args.strategy->get_output_rows() * args.depthwise_args.kernel_rows * args.depthwise_args.kernel_cols;
    }
    else
    {
      return sizeof(T *) * args.strategy->get_input_rows();
    }
  }

  static size_t sizeof_input_padding(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    // Round-up the number of columns to be a whole number of QUADS
    auto input_cols = arm_gemm::roundup<size_t>(args.strategy->get_input_cols(), 16 / sizeof(T));
    return sizeof(T) * input_cols;
  }

  static size_t sizeof_input_patch(const WorkspaceArgs<IDepthfirstStrategy, OutputStage> &args)
  {
    if (IsGeneric)
    {
      // Round-up the number of columns to be a whole number of QUADS
      auto output_cols = arm_gemm::roundup<size_t>(args.strategy->get_output_cols(), 16 / sizeof(T));
      const auto kernel_points = args.depthwise_args.kernel_rows * args.depthwise_args.kernel_cols;
      return sizeof(T) * kernel_points * args.strategy->get_output_rows() * output_cols;
    }
    else
    {
      // Round-up the number of columns to be a whole number of QUADS
      auto input_cols = arm_gemm::roundup<size_t>(args.strategy->get_input_cols(), 16 / sizeof(T));
      return sizeof(T) * args.strategy->get_input_rows() * input_cols;
    }
  }
};

template <bool IsGeneric, typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
struct StrategyType
{
  using Type = DepthfirstMultiplierStrategy<TInput, TWeight, TOutput, TAccum>;

  template <typename WorkspaceType>
  static void execute(
    const DepthwiseArgs &args, const WorkspaceType *ws, const Type *strat,
    const OutputStage &, const unsigned int,
    const void *parameters, const void *
  )
  {
    strat->get_kernel()(
      ws->input_rows,
      ws->outptr_array,
      parameters, args.channel_multiplier,
      ws->activation_min, ws->activation_max
    );
  }
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
struct StrategyType<true, TInput, TWeight, TOutput, TAccum, OutputStage>
{
  using Type = GenericDepthfirstMultiplierStrategy<TInput, TWeight, TOutput, TAccum, OutputStage>;

  template <typename WorkspaceType>
  static void execute(
    const DepthwiseArgs &args, const WorkspaceType *ws, const Type *strat,
    const OutputStage &, const unsigned int start_output_channel,
    const void *parameters, const void *bias
  )
  {
    strat->get_kernel()(
      ws->input_rows, ws->outptr_array,
      reinterpret_cast<const TWeight *>(parameters),
      bias == nullptr ? nullptr : reinterpret_cast<const TAccum *>(bias) + start_output_channel,
      strat->get_kernel_rows() * strat->get_kernel_cols(),
      args.channel_multiplier,
      ws->activation_min, ws->activation_max
    );
  }
};

template <typename TInput, typename TWeight, typename TOutput>
struct StrategyType<false, TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>
{
  using Type = DepthfirstMultiplierStrategy<TInput, TWeight, TOutput, int32_t>;

  template <typename WorkspaceType>
  static void execute(
    const DepthwiseArgs &args, const WorkspaceType *ws, const Type *strat,
    const arm_gemm::Requantize32 &qp, const unsigned int,
    const void *parameters, const void *
  )
  {
    strat->get_kernel()(
      ws->input_rows,
      ws->outptr_array,
      parameters, args.channel_multiplier,
      qp
    );
  }
};

template <typename TInput, typename TWeight, typename TOutput>
struct StrategyType<true, TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>
{
  using Type = GenericDepthfirstMultiplierStrategy<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>;

  template <typename WorkspaceType>
  static void execute(
    const DepthwiseArgs &args, const WorkspaceType *ws, const Type *strat,
    const arm_gemm::Requantize32 &qp, const unsigned int start_output_channel,
    const void *parameters, const void *
  )
  {
    auto get_ptr = [start_output_channel] (const int32_t *ptr) -> const int32_t *
    {
      return ptr == nullptr ? nullptr : ptr + start_output_channel;
    };

    strat->get_kernel()(
      ws->input_rows, ws->outptr_array,
      reinterpret_cast<const TWeight *>(parameters),
      get_ptr(qp.bias),
      strat->get_kernel_rows() * strat->get_kernel_cols(),
      args.channel_multiplier,
      get_ptr(qp.per_channel_left_shifts),
      get_ptr(qp.per_channel_muls),
      get_ptr(qp.per_channel_right_shifts),
      qp
    );
  }
};

template <bool IsGeneric> struct PrepareInputSample;

template <> struct PrepareInputSample<false>
{
  template <typename WorkspaceType, typename StrategyType, typename T>
  static void execute(
    const DepthwiseArgs &, WorkspaceType *ws, const StrategyType *strat,
    T *base_ptr, size_t ld_row, size_t ld_col,
    const unsigned int input_pad_top, const unsigned int valid_rows,
    const unsigned int input_pad_left, const unsigned int valid_cols
  )
  {
    fill_nchw_patch_array(
      ws->input_rows, ws->input_patch, strat->get_input_rows(), strat->get_input_cols(),
      base_ptr, ld_row, ld_col,
      ws->input_padding,
      input_pad_top, valid_rows,
      input_pad_left, valid_cols
    );
  }
};

template <> struct PrepareInputSample<true>
{
  template <typename WorkspaceType, typename StrategyType, typename T>
  static void execute(
    const DepthwiseArgs &args, WorkspaceType *ws, const StrategyType *strat,
    T *base_ptr, size_t ld_row, size_t ld_col,
    const unsigned int input_pad_top, const unsigned int valid_rows,
    const unsigned int input_pad_left, const unsigned int valid_cols
  )
  {
    fill_patch_array_generic_kernel(
      ws->input_rows, ws->input_patch,
      strat->get_output_rows(), strat->get_output_cols(),
      args.kernel_rows, args.kernel_cols,
      args.stride_rows, args.stride_cols,
      base_ptr, ld_row, ld_col,
      ws->input_padding,
      input_pad_top, valid_rows,
      input_pad_left, valid_cols
    );
  }
};

}  // namespace depthfirst_multiplier

template <typename TInput,
          typename TWeight=TInput,
          typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TInput>::Type,
          bool is_generic=false,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class DepthwiseDepthfirstMultiplier : public DepthfirstDriver<TInput, TWeight, TOutput>
{
  protected:
  using StratType = typename depthfirst_multiplier::StrategyType<is_generic, TInput, TWeight, TOutput, TAccum, OutputStage>::Type;
  using WorkspaceManager = Workspace<
    OutputArrayElement<TOutput>,
    depthfirst_multiplier::InputPatchElement<TInput, is_generic, OutputStage>,
    ActivationsElement<TOutput, OutputStage>
  >;
  using WorkingSpace = typename WorkspaceManager::WorkspaceType;

  OutputStage m_os;  // Copy of the output parameters
  const void *m_bias = nullptr;  // Copy of the bias (should we need it)

  public:
  DepthwiseDepthfirstMultiplier(StratType *const strat, const DepthwiseArgs &args, const OutputStage &os = {})
  : DepthfirstDriver<TInput, TWeight, TOutput>(strat, args), m_os(os)
  {
  }

  DepthwiseDepthfirstMultiplier(DepthwiseDepthfirstMultiplier &) = delete;
  DepthwiseDepthfirstMultiplier &operator=(DepthwiseDepthfirstMultiplier &) = delete;

  size_t get_storage_size(void) const override
  {
    return reinterpret_cast<const StratType *>(this->m_strat.get())
      ->get_storage_size(this->m_args);
  }

  void pack_parameters(void *buffer, const void *biases, const void *weights, size_t ld_weight_col, size_t ld_weight_row) override
  {
    reinterpret_cast<const StratType *>(this->m_strat.get())
      ->pack_parameters(this->m_args, buffer, biases, m_os, weights, ld_weight_col, ld_weight_row);
    m_bias = biases;
    depthwise_depthfirst::stash_bias(m_os, biases);
  }

  size_t get_working_size_per_thread(const unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    return WorkspaceManager::get_sizeof_workspace(WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, m_os));
  }

  void initialise_working_space(void *buffer, unsigned int n_input_channels) const override
  {
    DepthwiseArgs args(this->m_args);
    args.input_channels = n_input_channels;
    return WorkspaceManager::initialise(buffer, WorkspaceArgs<IDepthfirstStrategy, OutputStage>(this->m_strat.get(), args, m_os));
  }

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

    const int ii = static_cast<int>(output_i * args.stride_rows) - args.padding.top;
    const auto input_pad_top = static_cast<unsigned int>(ii < 0 ? -ii : 0);
    const auto input_i = static_cast<unsigned int>(ii < 0 ? 0 : ii);

    const int ij = static_cast<int>(output_j * args.stride_cols) - args.padding.left;
    const auto input_pad_left = static_cast<unsigned int>(ij < 0 ? -ij : 0);
    const auto input_j = static_cast<unsigned int>(ij < 0 ? 0 : ij);

    // Compute the output pointer array. We'll update this array after every
    // invocation of the kernel.
    fill_pointer_array(
      ws->outptr_array, this->m_strat->get_output_rows(), this->m_strat->get_output_cols(),
      output.base + output_i*output.ld_row + output_j*output.ld_col + output_channel_start,
      output.ld_row, output.ld_col,
      ws->output_buffer,
      0, args.output_rows - output_i, // Top padding, # valid rows
      0, args.output_cols - output_j  // Left padding, # valid columns
    );

    // Compute the parameter stride
    DepthwiseArgs single_iter(args);
    single_iter.input_channels = 1;
    const size_t parameter_stride = reinterpret_cast<const StratType *>(this->m_strat.get())
      ->get_storage_size(single_iter);

    for (; output_channel_start < output_channel_end;
         output_channel_start += args.channel_multiplier)
    {
      // Compute the input pointer array
      const auto input_channel = output_channel_start / args.channel_multiplier;

      // Construct the input patch
      depthfirst_multiplier::PrepareInputSample<is_generic>::execute(
        args, ws, this->m_strat.get(),
        input.base + input_channel + input_i*input.ld_row + input_j*input.ld_col, input.ld_row, input.ld_col,
        input_pad_top, args.input_rows - input_i,
        input_pad_left, args.input_cols - input_j
      );

      // Execute the kernel
      depthfirst_multiplier::StrategyType<is_generic, TInput, TWeight, TOutput, TAccum, OutputStage>::execute(
        args, ws, reinterpret_cast<const StratType *>(this->m_strat.get()), m_os, output_channel_start,
        parameters, m_bias
      );

      // Update the output pointers
      for (unsigned int n = 0; n < this->m_strat->get_output_rows() * this->m_strat->get_output_cols(); n++)
      {
        ws->outptr_array[n] += args.channel_multiplier;
      }

      // Progress the parameters
      parameters = reinterpret_cast<const char *>(parameters) + parameter_stride;
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
