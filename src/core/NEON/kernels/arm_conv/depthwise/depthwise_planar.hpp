/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "depthfirst_driver.hpp"
#include "interleaves/generic.hpp"

namespace arm_conv {
namespace depthwise {

template <typename OutputStage>
class IPlanarStrategy
{
  public:
  virtual ~IPlanarStrategy() = default;
  virtual unsigned int get_output_rows(void) const = 0;
  virtual arm_gemm::VLType get_vl_type(void) const = 0;

  virtual size_t get_storage_size(const DepthwiseArgs &) const = 0;
  virtual void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const OutputStage &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const = 0;
};


template <typename TInput, typename TWeight, typename TOutput, typename TAccum,
          typename OutputStage>
struct PlanarKernelType;

template <typename TInput, typename TWeight, typename TOutput, typename TAccum>
struct PlanarKernelType<TInput, TWeight, TOutput, TAccum, Nothing>
{
  using Type = std::function<void(
    const TInput *, size_t ld_in_row, size_t ld_in_col, size_t ld_in_vl,
    unsigned int pad_top, unsigned int valid_input_rows,
    unsigned int pad_left, unsigned int valid_input_cols,
    const TWeight *, const TAccum *,
    TOutput **, const size_t *, const size_t *, unsigned int output_cols,
    unsigned int start_channels, unsigned int valid_channels,
    TAccum act_min, TAccum act_max
  )>;

  template <typename WorkspaceType>
  static inline void execute(
    const Type fn,
    const TInput *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_in_vl,
    unsigned int pad_top, unsigned int valid_input_rows,
    unsigned int pad_left, unsigned int valid_input_cols,
    const TWeight *weights, const TAccum *bias,
    TOutput **outptrs, const size_t *outlds, const size_t *outvllds, unsigned int output_cols,
    unsigned int start_channel, unsigned int valid_channels,
    const Nothing &, const WorkspaceType *ws
  )
  {
    fn(
      inptr, ld_in_row, ld_in_col, ld_in_vl,
      pad_top, valid_input_rows,
      pad_left, valid_input_cols,
      weights, bias,
      outptrs, outlds, outvllds, output_cols,
      start_channel, valid_channels,
      ws->activation_min, ws->activation_max
    );
  }
};

template <typename TInput, typename TWeight, typename TOutput>
struct PlanarKernelType<TInput, TWeight, TOutput, int32_t, arm_gemm::Requantize32>
{
  using Type = std::function<void(
    const TInput *, size_t ld_in_row, size_t ld_in_col, size_t ld_in_vl,
    unsigned int pad_top, unsigned int valid_input_rows,
    unsigned int pad_left, unsigned int valid_input_cols,
    const TWeight *,
    TOutput **, const size_t *, const size_t *, unsigned int output_cols,
    unsigned int start_channel, unsigned int valid_channels,
    const arm_gemm::Requantize32 &
  )>;

  template <typename WorkspaceType>
  static inline void execute(
    const Type fn,
    const TInput *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_in_vl,
    unsigned int pad_top, unsigned int valid_input_rows,
    unsigned int pad_left, unsigned int valid_input_cols,
    const TWeight *weights, const int32_t *,
    TOutput **outptrs, const size_t *outlds, const size_t *outldvls, unsigned int output_cols,
    unsigned int first_channel, unsigned int valid_channels,
    const arm_gemm::Requantize32 &qp, const WorkspaceType *
  )
  {
    fn(
      inptr, ld_in_row, ld_in_col, ld_in_vl,
      pad_top, valid_input_rows,
      pad_left, valid_input_cols,
      weights,
      outptrs, outlds, outldvls, output_cols,
      first_channel, valid_channels,
      qp
    );
  }
};


template <typename TInput, typename TWeight=TInput, typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TOutput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class PlanarStrategy : public IPlanarStrategy<OutputStage>
{
  unsigned int m_kernel_rows, m_kernel_cols;
  unsigned int m_stride_rows, m_stride_cols;
  unsigned int m_output_rows;
  arm_gemm::VLType m_vl_type;

  protected:
  virtual bool get_kernel_packing_point(const unsigned int index, unsigned int &x, unsigned int &y) const
  {
    // Get the kernel point to pack at the given index; return false to
    // indicate that this index (and all greater indices) is out of range.
    if (m_kernel_rows * m_kernel_cols <= index)
      return false;

    y = index % m_kernel_cols;
    x = index / m_kernel_cols;
    return true;
  }

  virtual interleaves::PackingArguments get_kernel_packing_arguments(void) const
  {
    return interleaves::PackingArguments(
      m_kernel_rows, m_kernel_cols, sizeof(TWeight),
      false, sizeof(TAccum),  // Don't pack the bias
      m_vl_type, sizeof(TAccum), 1,  // Accumulator depth of 1 TODO
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
  }

  public:
  PlanarStrategy(
    unsigned int kernel_rows, unsigned int kernel_cols,
    unsigned int stride_rows, unsigned int stride_cols,
    unsigned int output_rows,
    arm_gemm::VLType vl_type
  ) : m_kernel_rows(kernel_rows), m_kernel_cols(kernel_cols),
      m_stride_rows(stride_rows), m_stride_cols(stride_cols),
      m_output_rows(output_rows), m_vl_type(vl_type)
  {
  }

  unsigned int get_output_rows(void) const override { return m_output_rows; }
  arm_gemm::VLType get_vl_type(void) const override { return m_vl_type; }

  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleaves::get_storage_size_generic(this->get_kernel_packing_arguments(), args);
  }

  void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const OutputStage &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const override
  {
    interleaves::pack_parameters_generic(
      this->get_kernel_packing_arguments(), args,
      buffer, biases, weights, ld_weight_col, ld_weight_row
    );
  }

  using KernelType = typename PlanarKernelType<TInput, TWeight, TOutput, TAccum, OutputStage>::Type;
  virtual KernelType get_kernel(void) const = 0;
};


namespace {

template <typename T>
struct OutputRowPtrsElement
{
  struct Workspace
  {
    T **output_row_ptrs;
    size_t *output_ld_cols;
    size_t *output_ld_vls;  // Stride between vectors of channels
    T *output_padding_buffer;
  };

  template <typename OutputStage>
  static size_t get_element_size(const WorkspaceArgs<IPlanarStrategy<OutputStage>, OutputStage> &args)
  {
    // We need one pointer and stride for each row of output, and an additional
    // blob of memory into which padded stores can go.
    return args.strategy->get_output_rows() * (sizeof(T *) + 2*sizeof(size_t)) +
           get_vector_length<char>(args.strategy->get_vl_type());
  }

  template <typename WorkspaceType, typename OutputStage>
  static void *initialise(WorkspaceType *ws, void *buffer,
                          const WorkspaceArgs<IPlanarStrategy<OutputStage>, OutputStage> &args)
  {
    const auto n_rows = args.strategy->get_output_rows();
    ws->output_row_ptrs = reinterpret_cast<T **>(buffer);
    ws->output_ld_cols = reinterpret_cast<size_t *>(ws->output_row_ptrs + n_rows);
    ws->output_ld_vls = ws->output_ld_cols + n_rows;
    ws->output_padding_buffer = reinterpret_cast<T *>(ws->output_ld_vls + n_rows);
    return ws->output_padding_buffer + get_vector_length<T>(args.strategy->get_vl_type());
  }
};

}  // namespace {anonymous}


template <typename TInput, typename TWeight=TInput, typename TOutput=TInput,
          typename TAccum=typename DefaultTAccum<TOutput>::Type,
          typename OutputStage=typename DefaultOutputStage<TOutput>::Type>
class DepthwisePlanar : public DepthwiseCommon<TInput, TWeight, TOutput>
{
  using Parent = DepthwiseCommon<TInput, TWeight, TOutput>;
  using StrategyType = IPlanarStrategy<OutputStage>;
  using WorkspaceManager = Workspace<
    OutputRowPtrsElement<TOutput>,
    ActivationsElement<TAccum, OutputStage>
  >;
  using WorkspaceType = typename WorkspaceManager::WorkspaceType;

  std::unique_ptr<StrategyType> m_strat;
  const TAccum *m_bias;
  OutputStage m_os;

  public:
  DepthwisePlanar(StrategyType *const strat, const DepthwiseArgs &args, const OutputStage &os = {})
  : Parent(args), m_strat(strat), m_bias(nullptr), m_os(os)
  {
  }

  DepthwisePlanar(DepthwisePlanar &) = delete;
  DepthwisePlanar &operator=(DepthwisePlanar &) = delete;

  size_t get_storage_size(void) const override
  {
    return m_strat->get_storage_size(this->m_args);
  }

  void pack_parameters(
    void *buffer, const void *biases,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) override
  {
    m_strat->pack_parameters(this->m_args, buffer, biases, {}, weights, ld_weight_col, ld_weight_row);
    this->m_bias = reinterpret_cast<const TAccum *>(biases);
    depthwise_depthfirst::stash_bias(this->m_os, biases);
  }

  size_t get_working_size(unsigned int n_threads, unsigned int) const override
  {
    return this->get_working_size_per_thread() * n_threads;
  }

  protected:
  /* Compute the amount of working space required for a single thread. */
  virtual size_t get_working_size_per_thread(void) const
  {
    return WorkspaceManager::get_sizeof_workspace(
      WorkspaceArgs<IPlanarStrategy<OutputStage>, OutputStage>(m_strat.get(), this->m_args, m_os));
  }

  /* Initialise the working space for a thread. */
  virtual void initialise_working_space(void *buffer) const
  {
    WorkspaceManager::initialise(
      buffer,
      WorkspaceArgs<IPlanarStrategy<OutputStage>, OutputStage>(m_strat.get(), this->m_args, m_os)
    );
  }

  /* Execute the kernel for a given chunk of work. */
  virtual void execute_kernel(
    const TInput *inptr, size_t ld_in_row, size_t ld_in_col, size_t ld_in_vl,
    unsigned int pad_top, unsigned int valid_input_rows,
    unsigned int pad_left, unsigned int valid_input_cols,
    const TWeight *weights, const TAccum *bias,
    TOutput *outptr, size_t ld_out_row, size_t ld_out_col, size_t ld_out_vl,
    unsigned int valid_output_rows, unsigned int valid_output_cols,
    unsigned int first_channel, unsigned int valid_channels,
    WorkspaceType *ws
  ) const
  {
    // Initialise the output pointers
    for (auto i = 0u; i < m_strat->get_output_rows(); i++)
    {
      // Point at the output tensor for all valid rows; otherwise point at the
      // padding buffer.
      ws->output_row_ptrs[i] = i < valid_output_rows ? outptr : ws->output_padding_buffer;
      ws->output_ld_cols[i] = i < valid_output_rows ? ld_out_col : 0;
      ws->output_ld_vls[i] = i < valid_output_rows ? ld_out_vl : 0;
      outptr += ld_out_row;
    }

    // Execute the kernel
    PlanarKernelType<TInput, TWeight, TOutput, TAccum, OutputStage>::template execute<WorkspaceType>(
      reinterpret_cast<const PlanarStrategy<TInput, TWeight, TOutput, TAccum, OutputStage> *>(m_strat.get())->get_kernel(),
      inptr, ld_in_row, ld_in_col, ld_in_vl,
      pad_top, valid_input_rows, pad_left, valid_input_cols,
      weights, bias,
      ws->output_row_ptrs, ws->output_ld_cols, ws->output_ld_vls,
      valid_output_cols, first_channel, valid_channels,
      this->m_os, ws
    );
  }

  void execute_internal(
    const DepthwiseArgs &args,
    const void *input,
    size_t ld_input_col,
    size_t ld_input_row,
    size_t ld_input_batch,
    const void *parameters,
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
      static_cast<uint8_t *>(working_space) + thread_id * this->get_working_size_per_thread();
    this->initialise_working_space(thread_working_space);
    auto ws = reinterpret_cast<WorkspaceType *>(thread_working_space);

    const auto n_output_channels = args.input_channels * args.channel_multiplier;
    const auto vl = get_vector_length<TAccum>(m_strat->get_vl_type());

    // Get typed pointers
    auto input_batch = reinterpret_cast<const TInput *>(input);
    auto output_batch = reinterpret_cast<TOutput *>(output);
    auto weights = reinterpret_cast<const TWeight *>(parameters);

    // Iterate over batches
    for (auto batches = args.n_batches; batches; batches--)
    {
      // NOTE: Other loop orderings are possible and it would be worth
      // investigating them.

      // Within a batch, stripe threads across rows.
      for (auto start_output_i = thread_id * m_strat->get_output_rows();
           start_output_i < args.output_rows;
           start_output_i += n_threads * m_strat->get_output_rows())
      {
        // Determine what (if any padding) is required on the top/bottom of
        // this row of the convolution.
        const int start_input_i = start_output_i * args.stride_rows - args.padding.top;
        const unsigned int input_pad_top = start_input_i < 0 ? -start_input_i : 0;
        const unsigned int input_i = start_input_i < 0 ? 0 : start_input_i;
        const unsigned int valid_input_rows = input_i > args.input_rows ? 0 : args.input_rows - input_i;
        const unsigned int valid_output_rows = args.output_rows - start_output_i;

        auto inptr_row = input_batch + input_i*ld_input_row;
        auto outptr_row = output_batch + start_output_i * ld_output_row;

        // Execute the kernel
        this->execute_kernel(
          inptr_row, ld_input_row, ld_input_col, vl,
          input_pad_top, valid_input_rows, args.padding.left, args.input_cols,
          weights, this->m_bias,
          outptr_row, ld_output_row, ld_output_col, vl,
          valid_output_rows, args.output_cols,
          0 /* first channel */, n_output_channels,
          ws
        );
      }

      // Update the input and output pointers to account for batch
      input_batch += ld_input_batch;
      output_batch += ld_output_batch;
    }
  }
};

}  // namespace depthwise
}  // namespace arm_conv
