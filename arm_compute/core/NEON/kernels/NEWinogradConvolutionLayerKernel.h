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
#ifndef __ARM_COMPUTE_NEGEMMWINOGRADCONVOLUTIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEGEMMWINOGRADCONVOLUTIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/convolution/common/convolution.hpp"
#include "arm_compute/core/NEON/kernels/convolution/common/tensor.hpp"
#include "arm_compute/core/NEON/kernels/convolution/winograd/batched_blocked_gemm.hpp"
#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_gemm.hpp"

namespace arm_compute
{
class ITensor;

/** Interface for the NEON kernel to perform Winograd input transform. */
template <typename T>
class INEWinogradLayerTransformInputKernel : public INEKernel
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
     *
     * @return Storage size (in units of TIn) required.
     */
    virtual unsigned int get_input_storage_size(int n_batches, int n_channels, int n_rows, int n_cols, bool same_padding) const = 0;

    /** Gets the stride between matrices in the input worspace
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] input_shape  The shape of the input tensor.
     * @param[in] padding_type The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const = 0;

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
    virtual void configure(const T *const input, const int n_batches, const int n_rows, const int n_cols, const int n_channels, const PaddingType padding, T *const output, const int matrix_stride) = 0;

    /** Destructor */
    virtual ~INEWinogradLayerTransformInputKernel()
    {
    }
};

/** NEON kernel to perform Winograd input transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformInputKernel : public INEWinogradLayerTransformInputKernel<T>
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
     *
     * @return Storage size (in units of TIn) required.
     */
    unsigned int get_input_storage_size(
        int  n_batches,
        int  n_channels,
        int  n_rows,
        int  n_cols,
        bool same_padding) const override;

    /** Gets the stride between matrices in the input worspace
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] input_shape  The shape of the input tensor.
     * @param[in] padding_type The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    int get_matrix_stride(const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const override;

    /** Default constructor */
    NEWinogradLayerTransformInputKernel();

    const char *name() const override
    {
        return "NEWinogradLayerTransformInputKernel";
    }

    /** Configure the output transform kernel.
     *
     * @param[in]  input         Input tensor data. Data types supported: F32.
     * @param[in]  n_batches     Number of batches in input tensor.
     * @param[in]  n_rows        Number of rows in input tensor.
     * @param[in]  n_cols        Number of columns in input tensor.
     * @param[in]  n_channels    Number of channels in input tensor.
     * @param[in]  padding       Padding type.
     * @param[out] output        Base of output matrices.
     * @param[in]  matrix_stride Stride between output matrices.
     */
    void configure(
        const T *const    input,
        const int         n_batches,
        const int         n_rows,
        const int         n_cols,
        const int         n_channels,
        const PaddingType padding,
        T *const          output,
        const int         matrix_stride) override;

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

    /** Winograd base kernel */
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelCols, KernelCols>;
    /** Winograd convolution kernel */
    using WinogradConv = typename WinogradBase::template Convolution<T, T>;

    /** Static function to check if given info will lead to a valid configuration of @ref NEWinogradLayerTransformInputKernel
     *
     * @param[in] input         First tensor input info. Data types supported: F32.
     * @param[in] output        Output tensor info. Data types supported: same as @p input.
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);

private:
    using InputTransform = typename WinogradBase::template InputTransform<T>;
    std::unique_ptr<InputTransform> _transform;
};

/** Interface for the NEON kernel to perform Winograd output transform. */
template <typename T>
class INEWinogradLayerTransformOutputKernel : public INEKernel
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
     *
     * @return Storage size (in units of TOut) required.
     */
    virtual unsigned int get_output_storage_size(int n_batches, int n_rows, int n_cols, int n_output_channels, bool same_padding) const = 0;

    /** Gets the stride between matrices in the output worspace
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] input_shape  The shape of the input tensor.
     * @param[in] padding_type The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const = 0;

    /** Get the output shape of a convolution.
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] in_shape     The shape of the input tensor.
     * @param[in] padding      The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    virtual Tensor4DShape get_output_shape(const KernelShape &kernel_shape, const Tensor4DShape &in_shape, const PaddingType padding) const = 0;

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
    virtual void configure(
        const ITensor *biases,
        const T *const output_workingspace,
        const int      matrix_stride,
        T *const       output,
        const int      n_batches,
        const int      n_rows,
        const int      n_cols,
        const int      n_channels) = 0;

    virtual ~INEWinogradLayerTransformOutputKernel()
    {
    }
};

/** NEON kernel to perform Winograd output transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformOutputKernel : public INEWinogradLayerTransformOutputKernel<T>
{
public:
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
    /** Default destructor */
    ~NEWinogradLayerTransformOutputKernel() = default;

    // Inherited methods overridden:
    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in] n_batches         Number of batches in the output tensor.
     * @param[in] n_rows            Number of rows in each feature map of the input tensor.
     * @param[in] n_cols            Number of columns in each feature map of the input tensor.
     * @param[in] n_output_channels Number of feature maps in the output tensor.
     * @param[in] same_padding      Use "SAME" padding, otherwise use "VALID".
     *
     * @return Storage size (in units of TOut) required.
     */
    unsigned int get_output_storage_size(int n_batches, int n_rows, int n_cols, int n_output_channels, bool same_padding) const override;

    /** Gets the stride between matrices in the output worspace
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] input_shape  The shape of the input tensor.
     * @param[in] padding_type The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    int get_matrix_stride(const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const override;
    /** Get the output shape of a convolution.
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     * @param[in] in_shape     The shape of the input tensor.
     * @param[in] padding      The type of padding to be used.
     *
     * @return Stride expressed in bytes.
     */
    Tensor4DShape get_output_shape(const KernelShape &kernel_shape, const Tensor4DShape &in_shape, const PaddingType padding) const override;

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
        const ITensor *biases,
        const T *const output_workingspace,
        const int      matrix_stride,
        T *const       output,
        const int      n_batches,
        const int      n_rows,
        const int      n_cols,
        const int      n_channels) override;

    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

    /** Static function to check if given info will lead to a valid configuration of @ref NEWinogradLayerTransformOutputKernel
     *
     * @param[in]  input         Source tensor with shape [C, N, 16, batches] or [C, N, 36, batches]. Data types supported: F32.
     * @param[in]  bias          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. It can be a nullptr. Data type supported: as @p input
     * @param[out] output        Destination tensor with shape [output_convolved_dims.width, output_convolved_dims.height, C, batches]. Data type supported: same as @p input
     * @param[in]  winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info);

private:
    using WinogradBase    = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WinogradConv    = typename WinogradBase::template Convolution<T, T>;
    using OutputTransform = typename WinogradBase::template OutputTransform<T>;

    const ITensor *_biases;
    const T       *_output_workspace;
    int            _matrix_stride;
    int            _matrix_row_stride;
    T             *_output;
    int            _n_batches;
    int            _n_rows;
    int            _n_cols;
    int            _n_channels;
};

/** Interface for the NEON kernel to perform Winograd weights transform. */
template <typename T>
class INEWinogradLayerTransformWeightsKernel : public INEKernel
{
public:
    /** Determine how much memory (in units of T) to allocate for the
     * transformed weights.
     *
     * @param[in] n_output_channels Number of output feature maps.
     * @param[in] n_input_channels  Number of input feature maps.
     *
     * @return Storage size (in units of T) required.
     */
    virtual unsigned int get_weight_storage_size(int n_output_channels, int n_input_channels) const = 0;
    /** Gets the stride between matrices in the kernel worspace
     *
     * @param[in] kernel_shape The shape of the weights tensor.
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(const KernelShape &kernel_shape) const = 0;

    /** Configure the weights transform kernel.
     *
     * @param[in] weights_hwio      Pointer to the weights tensor
     * @param[in] output            Pointer to working space for the output tensor in the Winograd domain.
     * @param[in] matrix_stride     Stride across matrices in the output workspace.
     * @param[in] n_output_channels Number of filters.
     * @param[in] n_input_channels  Number of channels in each filter.
     */
    virtual void configure(const ITensor *weights_hwio, T *const output, const int matrix_stride, const int n_output_channels, const int n_input_channels) = 0;

    virtual ~INEWinogradLayerTransformWeightsKernel()
    {
    }
};

/** NEON kernel to perform Winograd weights transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerTransformWeightsKernel final : public INEWinogradLayerTransformWeightsKernel<T>
{
public:
    /** Default constructor. */
    NEWinogradLayerTransformWeightsKernel();
    const char *name() const override
    {
        return "NEWinogradLayerTransformWeightsKernel";
    }

    /** Static function to check if given info will lead to a valid configuration of @ref NEWinogradLayerTransformWeightsKernel
     *
     * @param[in] input         Source tensor info. The input is a 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] (NCHW data layout).
     *                          kernel_x must be 3 and equal to kernel_y. Data types supported: F32.
     * @param[in] output        Destination tensor info. The output is a 3D tensor with dimensions [OFM, IFM, 16] or [OFM, IFM, 36]. Data type supported: same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);

    // Inherited methods overridden:
    void configure(const ITensor *weights_hwio, T *const output, const int matrix_stride, const int n_output_channels, const int n_input_channels) override;
    unsigned int get_weight_storage_size(int n_output_channels, int n_input_channels) const override;
    int get_matrix_stride(const KernelShape &kernel_shape) const override;
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    using WinogradBase     = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using WinogradConv     = typename WinogradBase::template Convolution<T, T>;
    using WeightsTransform = typename WinogradBase::template WeightsTransform<T>;
    std::unique_ptr<WeightsTransform> _transform;
};

/** Interface for the NEON kernel to perform Winograd. */
template <typename TIn, typename TOut>
class INEWinogradLayerBatchedGEMMKernel : public INEKernel
{
public:
    /** Get the number of GEMMs to compute
     */
    virtual unsigned int get_number_gemms() const = 0;
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
    virtual void configure(
        const unsigned int n_gemms,
        const int M, const int K, const int N,
        const int        a_matrix_stride,
        const int        a_row_stride,
        const int        b_matrix_stride,
        const int        b_row_stride,
        const int        c_matrix_stride,
        const int        c_row_stride,
        const TIn *const a_ptr,
        const TIn *const b_ptr,
        TOut *const      c_ptr) = 0;

    /** Get the number of tiles per row
     */
    virtual int get_output_tile_rows() const = 0;
    /** Get the number of tiles per columns
     */
    virtual int get_output_tile_cols() const = 0;
    /** Get the number of blocks
     */
    virtual int get_number_blocks() const = 0;
};

/** NEON kernel to perform Winograd. */
template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class NEWinogradLayerBatchedGEMMKernel : public INEWinogradLayerBatchedGEMMKernel<TIn, TOut>
{
public:
    /** Winograd base kernel */
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    /** Winograd convolution kernel */
    using WinogradConv = typename WinogradBase::template Convolution<TIn, TOut>;
    /** Winograd batched blocked GEMM operator */
    using MultiGEMM = winograd::BatchedBlockedGemm<WinogradConv::M_BLOCK, WinogradConv::N_BLOCK, TIn, TOut>;

    const char *name() const override
    {
        return "NEWinogradLayerBatchedGEMMKernel";
    }
    /** Constructor */
    NEWinogradLayerBatchedGEMMKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerBatchedGEMMKernel(const NEWinogradLayerBatchedGEMMKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEWinogradLayerBatchedGEMMKernel &operator=(const NEWinogradLayerBatchedGEMMKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEWinogradLayerBatchedGEMMKernel(NEWinogradLayerBatchedGEMMKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEWinogradLayerBatchedGEMMKernel &operator=(NEWinogradLayerBatchedGEMMKernel &&) = default;
    /** Default destructor. */
    ~NEWinogradLayerBatchedGEMMKernel() = default;

    // Inherited methods overridden:

    unsigned int get_number_gemms() const override;
    int          get_output_tile_rows() const override;
    int          get_output_tile_cols() const override;
    int          get_number_blocks() const override;

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
        const int        a_matrix_stride,
        const int        a_row_stride,
        const int        b_matrix_stride,
        const int        b_row_stride,
        const int        c_matrix_stride,
        const int        c_row_stride,
        const TIn *const a_ptr,
        const TIn *const b_ptr,
        TOut *const      c_ptr) override;

    void run(const Window &window, const ThreadInfo &info) override;

    /** Static function to check if given info will lead to a valid configuration of @ref NEWinogradLayerBatchedGEMMKernel.
     *
     * @param[in]  a         First input tensor  (Matrix or Vector A). Data types supported: F32
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output    Output tensor. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensor *c, const ITensorInfo *output, const float alpha, const float beta, const GEMMInfo &gemm_info = GEMMInfo());

private:
    static const int           _output_tile_rows = OutputTileRows;
    static const int           _output_tile_cols = OutputTileCols;
    std::unique_ptr<MultiGEMM> _gemms;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMWINOGRADCONVOLUTIONLAYERKERNEL_H__*/
