/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/winograd/batched_blocked_gemm.hpp"
#include "arm_compute/core/NEON/kernels/winograd/convolution.hpp"
#include "arm_compute/core/NEON/kernels/winograd/tensor.hpp"
#include "arm_compute/core/NEON/kernels/winograd/winograd_gemm.hpp"

namespace arm_compute
{
class ITensor;

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformInputKernel : public INEKernel
{
public:
    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     *
     * @param[in] n_batches    Number of batches in the input tensor.
     * @param[in] n_channels   Number of feature maps in the input tensor.
     * @param[in] n_rows       Number of rows in each feature map.
     * @param[in] n_cols       Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_input_storage_size(
        int  n_batches,
        int  n_channels,
        int  n_rows,
        int  n_cols,
        bool same_padding);

    NEWinogradLayerTransformInputKernel();
    const char *name() const override
    {
        return "NEWinogradLayerTransformInputKernel";
    }

    /** Configure the output transform kernel.
     *
     * @param[in]  input         Input tensor data
     * @param[in]  n_batches     Number of batches in input tensor.
     * @param[in]  n_rows        Number of rows in input tensor.
     * @param[in]  n_cols        Number of columns in input tensor.
     * @param[in]  n_channels    Number of channels in input tensor.
     * @param[in]  padding       Padding type.
     * @param[out] output        Base of output matrices.
     * @param[in]  matrix_stride Stride between output matrices.
     */
    void configure(
        const float *const input,
        const int          n_batches,
        const int          n_rows,
        const int          n_cols,
        const int          n_channels,
        const PaddingType  padding,
        float *const       output,
        const int          matrix_stride);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    using WinogradBase   = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelCols, KernelCols>;
    using WinogradConv   = typename WinogradBase::template Convolution<float, float>;
    using InputTransform = typename WinogradBase::template InputTransform<float>;
    std::unique_ptr<InputTransform> _transform;
};

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformOutputKernel : public INEKernel
{
public:
    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in] n_batches         Number of batches in the output tensor.
     * @param[in] n_rows            Number of rows in each feature map of the input tensor.
     * @param[in] n_cols            Number of columns in each feature map of the input tensor.
     * @param[in] n_output_channels Number of feature maps in the output tensor.
     * @param[in] same_padding      Use "SAME" padding, otherwise use "VALID".
     */
    static unsigned int get_output_storage_size(
        int  n_batches,
        int  n_rows,
        int  n_cols,
        int  n_output_channels,
        bool same_padding);

    const char *name() const override
    {
        return "NEWinogradLayerTransformOutputKernel";
    }
    /** Constructor */
    NEWinogradLayerTransformOutputKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerTransformOutputKernel(const NEWinogradLayerTransformOutputKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerTransformOutputKernel &operator=(const NEWinogradLayerTransformOutputKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerTransformOutputKernel(NEWinogradLayerTransformOutputKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerTransformOutputKernel &operator=(NEWinogradLayerTransformOutputKernel &&) = default;

    ~NEWinogradLayerTransformOutputKernel() = default;

    /** Configure the output transform kernel.
     *
     * @param[in]  biases              Pointer to the biases tensor.
     * @param[in]  output_workingspace Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride       Output matrix stride, can be computed with winograd::WinogradGEMM<2, 2, 3, 3>::Convolution<float, float>::get_output_matrix_stride()
     * @param[out] output              Pointer to NHWC ordered output tensor, in the spatial domain.
     * @param[in]  n_batches           Number of batches in the input tensor.
     * @param[in]  n_rows              Number of rows in output tensor.
     * @param[in]  n_cols              Number of columns in output tensor.
     * @param[in]  n_channels          Number of feature maps in the output tensor.
     */
    void configure(
        const ITensor     *biases,
        const float *const output_workingspace,
        const int          matrix_stride,
        float *const       output,
        const int          n_batches,
        const int          n_rows,
        const int          n_cols,
        const int          n_channels);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    using WinogradBase    = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WinogradConv    = typename WinogradBase::template Convolution<float, float>;
    using OutputTransform = typename WinogradBase::template OutputTransform<float>;

    const ITensor *_biases;
    const float   *_output_workspace;
    int            _matrix_stride;
    int            _matrix_row_stride;
    float         *_output;
    int            _n_batches;
    int            _n_rows;
    int            _n_cols;
    int            _n_channels;
};

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformWeightsKernel final : public INEKernel
{
public:
    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed weights.
     *
     * @param[in] n_output_channels Number of output feature maps.
     * @param[in] n_input_channels  Number of input feature maps.
     */
    static unsigned int get_weight_storage_size(int n_output_channels, int n_input_channels);

    NEWinogradLayerTransformWeightsKernel();
    const char *name() const override
    {
        return "NEWinogradLayerTransformWeightsKernel";
    }
    /** Configure the output transform kernel.
     *
     * @param[in] weights_hwio      Pointer to the weights tensor
     * @param[in] output            Pointer to working space for the output tensor in the Winograd domain.
     * @param[in] matrix_stride     Stride across matrices in the output workspace.
     * @param[in] n_output_channels Number of filters.
     * @param[in] n_input_channels  Number of channels in each filter.
     */
    void configure(
        const ITensor *weights_hwio,
        float *const   output,
        const int      matrix_stride,
        const int      n_output_channels,
        const int      n_input_channels);

    // Inherited methods overridden:

    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    using WinogradBase     = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WinogradConv     = typename WinogradBase::template Convolution<float, float>;
    using WeightsTransform = typename WinogradBase::template WeightsTransform<float>;
    std::unique_ptr<WeightsTransform> _transform;
};

template <int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerKernel : public INEKernel
{
public:
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WinogradConv = typename WinogradBase::template Convolution<float, float>;
    using MultiGEMM    = winograd::BatchedBlockedGemm<WinogradConv::M_BLOCK, WinogradConv::N_BLOCK, float, float>;

    static const int _output_tile_rows = OutputTileRows;
    static const int _output_tile_cols = OutputTileCols;

    const char *name() const override
    {
        return "NEWinogradLayerKernel";
    }
    /** Constructor */
    NEWinogradLayerKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel(const NEWinogradLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerKernel &operator=(const NEWinogradLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel(NEWinogradLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerKernel &operator=(NEWinogradLayerKernel &&) = default;

    ~NEWinogradLayerKernel() = default;

    /** Initialise the kernel
     *
     * @param[in]  n_gemms         Number of GEMMs to compute.
     * @param[in]  M               in_shape.n_batches * tile_rows * tile_cols.
     * @param[in]  K               Number of channels in the input tensor.
     * @param[in]  N               Number of channels in the output tensor.
     * @param[in]  a_matrix_stride Stride between input matrices.
     * @param[in]  a_row_stride    Row stride inside input matrix.
     * @param[in]  b_matrix_stride Stride between weights matrices.
     * @param[in]  b_row_stride    Row stride inside the weights matrix.
     * @param[in]  c_matrix_stride Stride between output matrices.
     * @param[in]  c_row_stride    Row stride inside the output matrix.
     * @param[out] a_ptr           Input workspace.
     * @param[out] b_ptr           Kernel workspace.
     * @param[out] c_ptr           Output workspace.
     */
    void configure(
        const unsigned int n_gemms,
        const int M, const int K, const int N,
        const int          a_matrix_stride,
        const int          a_row_stride,
        const int          b_matrix_stride,
        const int          b_row_stride,
        const int          c_matrix_stride,
        const int          c_row_stride,
        const float *const a_ptr,
        const float *const b_ptr,
        float *const       c_ptr);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    std::unique_ptr<MultiGEMM> _gemms;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMWINOGRADLAYERKERNEL_H__*/
