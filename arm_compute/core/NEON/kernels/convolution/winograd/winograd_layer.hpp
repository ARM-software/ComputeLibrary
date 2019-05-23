/*
 * Copyright (c) 2019 ARM Limited.
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

#include <utility>

#include "arm_gemm_local.hpp"
#include "arm_gemm.hpp"
#include "winograd.hpp"

namespace winograd
{


class IWinogradConvolutionLayer
{
  public:
    virtual ~IWinogradConvolutionLayer() = default;

    virtual unsigned int weight_transform_get_window(void) const = 0;
    virtual void weight_transform_run(unsigned int start, unsigned int stop) = 0;

    virtual ITransform& input_transform(void) = 0; // Expose the input transform
    virtual ITransform& output_transform(void) = 0;  // Expose the output transform
    virtual arm_gemm::IGemmCommon *gemm(void) = 0;  // Expose the underlying GEMM
};

/** Example of how to construct an ACL-like interface.
 *
 * Use `get_weight_storage_size`, `get_input_storage_size` and
 * `get_output_storage_size` to allocate memory for the convolution engine.
 * Then create a `WinogradConvolutionLayer`.
 *
 * Initialise the weights using `weights_transform.run(...)`.
 *
 * For each inference:
 *   1. Transform the inputs to the Winograd domain using `input_transform.run(...)`
 *   2. Perform a number of GEMMs using `gemms.run(...)`
 *   3. Transform the output to the spatial domain using `output_transform.run(...)`
 */
template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols,
          typename TIn, typename TInGEMM, typename TOutGEMM, typename TOut,
          WinogradRoots Roots>
class WinogradConvolutionLayer : public IWinogradConvolutionLayer
{
  private:
    static constexpr int InnerTileRows = OutputTileRows + KernelRows - 1;
    static constexpr int InnerTileCols = OutputTileCols + KernelCols - 1;
    static constexpr int N_GEMMS = InnerTileRows * InnerTileCols;

    const KernelShape _kernel_shape;
    const Tensor4DShape _input_shape;
    const PaddingType _padding;
    const Tensor4DShape _output_shape;
    const int _n_output_rows, _n_output_cols;
    const int _kernel_matrix_stride, _kernel_matrix_row_stride;
    const int _input_matrix_stride, _input_matrix_row_stride;
    const int _output_matrix_stride, _output_matrix_row_stride;
    const int _tile_rows, _tile_cols;
    const int _m, _k, _n;

  public:
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols, Roots>;
    using WeightsTransform = typename WinogradBase::template WeightsTransform<TIn, TInGEMM>;
    using InputTransform = typename WinogradBase::template InputTransform<TIn, TInGEMM>;
    using WinogradConv = typename WinogradBase::template Convolution<TOut, TIn, TInGEMM, TOutGEMM>;
    using OutputTransform = typename WinogradBase::template OutputTransform<TOutGEMM, TOut>;

    /* Public member variables. */
    WeightsTransform weights_transform;  /** Operator to transform weights to Winograd domain. */
    InputTransform _input_transform;      /** Operator to transform input to Winograd domain. */
    arm_gemm::UniqueGemmCommon<TInGEMM, TOutGEMM> gemms;    /** Operator to perform multiple GEMMs. */
    OutputTransform _output_transform;    /** Operator to transform output from Winograd domain. */

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed weights.
     */
    static unsigned int get_weight_storage_size(
      const int n_output_channels,  /** Number of output feature maps. */
      const int n_input_channels    /** Number of input feature maps. */
    );

    static unsigned int get_weight_stride(
      const int n_output_channels,  /** Number of output feature maps. */
      const int n_input_channels    /** Number of input feature maps. */
    );

    static unsigned int get_weight_multi_stride(
      const int n_output_channels,  /** Number of output feature maps. */
      const int n_input_channels    /** Number of input feature maps. */
    );

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     */
    static unsigned int get_input_storage_size(
      const int n_batches,     /** Number of batches in the input tensor. */
      const int n_channels,    /** Number of feature maps in the input tensor. */
      const int n_rows,        /** Number of rows in each feature map. */
      const int n_cols,        /** Number of columns in each feature map. */
      const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
    );

    /** Get the row stride for the A matrix in the Winograd domain. */
    static unsigned int get_input_stride(
      const int n_batches,     /** Number of batches in the input tensor. */
      const int n_channels,    /** Number of feature maps in the input tensor. */
      const int n_rows,        /** Number of rows in each feature map. */
      const int n_cols,        /** Number of columns in each feature map. */
      const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
    );

    /** Get the stride between A matrices in the Winograd domain. */
    static unsigned int get_input_multi_stride(
      const int n_batches,     /** Number of batches in the input tensor. */
      const int n_channels,    /** Number of feature maps in the input tensor. */
      const int n_rows,        /** Number of rows in each feature map. */
      const int n_cols,        /** Number of columns in each feature map. */
      const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
    );

    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     */
    static unsigned int get_output_storage_size(
      const int n_batches,          /** Number of batches in the output tensor. */
      const int n_rows,             /** Number of rows in each feature map of the input tensor. */
      const int n_cols,             /** Number of columns in each feature map of the input tensor. */
      const int n_output_channels,  /** Number of feature maps in the output tensor. */
      const bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
    );

    static unsigned int get_output_stride(
      const int n_batches,          /** Number of batches in the output tensor. */
      const int n_rows,             /** Number of rows in each feature map of the input tensor. */
      const int n_cols,             /** Number of columns in each feature map of the input tensor. */
      const int n_output_channels,  /** Number of feature maps in the output tensor. */
      const bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
    );

    static unsigned int get_output_multi_stride(
      const int n_batches,          /** Number of batches in the output tensor. */
      const int n_rows,             /** Number of rows in each feature map of the input tensor. */
      const int n_cols,             /** Number of columns in each feature map of the input tensor. */
      const int n_output_channels,  /** Number of feature maps in the output tensor. */
      const bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
    );

    /** Get the shape (rows, cols) of a feature map of the output tensor. */
    static std::pair<int, int> get_output_feature_map_shape(
      const int n_input_rows,  /** Number of rows in the input feature map. */
      const int n_input_cols,  /** Number of columns in the input feature map. */
      const bool same_padding  /** Use "SAME" padding, otherwise use "VALID". */
    );

    /** Create a new Winograd convolution layer.
     */
    WinogradConvolutionLayer(
      const arm_gemm::CPUInfo &cpuinfo,       /** Describes CPU properties. */
      const int n_threads,          /** Maximum number of threads used to execute the convolution. */
      const int n_batches,          /** Number of batches in the input and output tensors. */
      const int n_input_channels,   /** Number of feature maps in a batch of the input tensor. */
      const int n_input_rows,       /** Number of rows in a feature map of the input tensor. */
      const int n_input_cols,       /** Number of columns in a feature map of the input tensor. */
      const int n_output_channels,  /** Number of feature maps in the output tensor. */
      const bool same_padding,      /** Use "SAME" padding, otherwise use "VALID". */
      const TIn* const weights,     /** Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps. */
      TInGEMM* const weights_storage,  /** Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size`. */
      const TIn* const input,       /** Pointer to NHWC ordered input tensor, in the spatial domain. */
      TInGEMM* const winograd_input,    /** Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`. */
      const TOut* const biases,     /** Pointer to biases vector. Pass nullptr if no bias is provided. */
      TOut* const output,           /** Pointer to NHWC ordered output tensor, in the spatial domain. */
      TOutGEMM* const winograd_output,  /** Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`. */
      const bool pretranspose_B=true,         /** Hint that the B matrix can be pretransposed. */
      arm_gemm::GemmConfig *gemm_cfg=nullptr  /** Pointer to GEMM configuration. */
    );

    /* Utility methods for interacting with the layer. */
    unsigned int weight_transform_get_window(void) const;
    void weight_transform_run(const unsigned int start, const unsigned int stop);

    ITransform& input_transform(void);
    ITransform& output_transform(void);

    /* Get a pointer to the GEMM underlying the Winograd transform. */
    arm_gemm::IGemmCommon *gemm(void);
};

}
