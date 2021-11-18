/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H
#define ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H

#include "src/core/NEON/kernels/convolution/common/convolution.hpp"
#include "src/core/NEON/kernels/convolution/common/tensor.hpp"
#include "src/cpu/ICpuKernel.h"

#include "src/core/NEON/kernels/convolution/winograd/winograd_layer.hpp"

namespace arm_compute
{
namespace cpu
{
/** Interface for the kernel to perform Winograd input transform. */
class ICpuWinogradConv2dTransformInputKernel : public NewICpuKernel<ICpuWinogradConv2dTransformInputKernel>
{
public:
    /** Get the working space required to perform the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param num_threads The greatest number of threads that will be used to execute the transform.
     * @return Size of working space required in bytes.
     */
    virtual unsigned int get_working_space_size(unsigned int num_threads) const = 0;

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     *
     * @param[in] num_batches  Number of batches in the input tensor.
     * @param[in] num_channels Number of feature maps in the input tensor.
     * @param[in] num_rows     Number of rows in each feature map.
     * @param[in] num_cols     Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     *
     * @return Storage size (in units of TIn) required.
     */
    virtual unsigned int get_input_storage_size(int num_batches, int num_channels, int num_rows, int num_cols, bool same_padding) const = 0;

    /** Gets the stride between matrices in the input worspace
     *
     * @param[in] num_batches  Number of batches in the input tensor.
     * @param[in] num_channels Number of feature maps in the input tensor.
     * @param[in] num_rows     Number of rows in each feature map.
     * @param[in] num_cols     Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(int num_batches, int num_channels, int num_rows, int num_cols, bool same_padding) const = 0;

    /** Configure the output transform kernel.
     *
     * @param[in]  input_nhwc    Input tensor in NHWC data layout format.
     * @param[in]  num_batches   Number of batches in input tensor.
     * @param[in]  num_rows      Number of rows in input tensor.
     * @param[in]  num_cols      Number of columns in input tensor.
     * @param[in]  num_channels  Number of channels in input tensor.
     * @param[in]  padding       Padding type.
     * @param[out] output        Base of output matrices.
     * @param[in]  matrix_stride Stride between output matrices.
     * @param[in]  workspace     Tensor to be used as the working space during the computation.
     */
    virtual void configure(const ITensorInfo *input_nhwc, const int num_batches, const int num_rows, const int num_cols, const int num_channels,
                           const PaddingType padding, ITensorInfo *output, const int matrix_stride, ITensorInfo *workspace) = 0;

    /** Destructor */
    virtual ~ICpuWinogradConv2dTransformInputKernel()
    {
    }
};

/** Kernel to perform Winograd input transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class CpuWinogradConv2dTransformInputKernel : public ICpuWinogradConv2dTransformInputKernel
{
public:
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformInputKernel(const CpuWinogradConv2dTransformInputKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformInputKernel &operator=(const CpuWinogradConv2dTransformInputKernel &) = delete;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformInputKernel(CpuWinogradConv2dTransformInputKernel &&) = default;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformInputKernel &operator=(CpuWinogradConv2dTransformInputKernel &&) = default;
    /** Default destructor */
    ~CpuWinogradConv2dTransformInputKernel() = default;

    /** Determine how much memory (in units of TIn) to allocate for the
     * transformed input.
     *
     * @param[in] num_batches  Number of batches in the input tensor.
     * @param[in] num_channels Number of feature maps in the input tensor.
     * @param[in] num_rows     Number of rows in each feature map.
     * @param[in] num_cols     Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     *
     * @return Storage size (in units of TIn) required.
     */
    unsigned int get_input_storage_size(
        int  num_batches,
        int  num_channels,
        int  num_rows,
        int  num_cols,
        bool same_padding) const override;

    /** Get the working space required to perform the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param[in] num_threads The greatest number of threads that will be used to execute the transform.
     *
     * @return Size of working space required in bytes.
     */
    unsigned int get_working_space_size(unsigned int num_threads) const override;

    /** Gets the stride between matrices in the input worspace
     *
     * @param[in] num_batches  Number of batches in the input tensor.
     * @param[in] num_channels Number of feature maps in the input tensor.
     * @param[in] num_rows     Number of rows in each feature map.
     * @param[in] num_cols     Number of columns in each feature map.
     * @param[in] same_padding Use "SAME" padding, otherwise use "VALID".
     *
     * @return Stride expressed in bytes.
     */
    int get_matrix_stride(
        int  num_batches,
        int  num_channels,
        int  num_rows,
        int  num_cols,
        bool same_padding) const override;

    /** Default constructor */
    CpuWinogradConv2dTransformInputKernel();

    const char *name() const override
    {
        return "CpuWinogradConv2dTransformInputKernel";
    }

    /** Configure the output transform kernel.
     *
     * @param[in]  input_nhwc    Input tensor.  Data types supported: F16/F32. Layout supported NHWC.
     * @param[in]  num_batches   Number of batches in input tensor.
     * @param[in]  num_rows      Number of rows in input tensor.
     * @param[in]  num_cols      Number of columns in input tensor.
     * @param[in]  num_channels  Number of channels in input tensor.
     * @param[in]  padding       Padding type.
     * @param[out] output        Base of output matrices.
     * @param[in]  matrix_stride Stride between output matrices.
     * @param[in]  workspace     Tensor to be used as the working space during the computation.
     */
    void configure(
        const ITensorInfo *input_nhwc,
        const int          num_batches,
        const int          num_rows,
        const int          num_cols,
        const int          num_channels,
        const PaddingType  padding,
        ITensorInfo       *output,
        const int          matrix_stride,
        ITensorInfo       *workspace) override;

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    /** Winograd base kernel */
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols, winograd::WinogradRoots::Integers>;
    /** Winograd convolution kernel */
    using WinogradConv = typename WinogradBase::template Convolution<T, T>;

    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2dTransformInputKernel
     *
     * @param[in] input         First tensor input info. Data types supported: F16/F32.
     * @param[in] output        Output tensor info. Data types supported: same as @p input.
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);

private:
    using InputTransform = typename WinogradBase::template InputTransform<T, T>;

    std::unique_ptr<InputTransform> _transform{ nullptr };
    int                             _num_channels;  /**< Number of channels in input tensor. */
    int                             _matrix_stride; /**< Stride between output matrices. */
};

/** Interface for the kernel to perform Winograd output transform. */
class ICpuWinogradConv2dTransformOutputKernel : public NewICpuKernel<ICpuWinogradConv2dTransformOutputKernel>
{
public:
    /** Get the working space required to perform the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param[in] num_threads The greatest number of threads that will be used to execute the transform.
     *
     * @return Size of working space required in bytes.
     */
    virtual unsigned int get_working_space_size(unsigned int num_threads) const = 0;

    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in] num_batches         Number of batches in the output tensor.
     * @param[in] num_rows            Number of rows in each feature map of the input tensor.
     * @param[in] num_cols            Number of columns in each feature map of the input tensor.
     * @param[in] num_output_channels Number of feature maps in the output tensor.
     *
     * @return Storage size (in units of TOut) required.
     */
    virtual unsigned int get_output_storage_size(int num_batches, int num_rows, int num_cols, int num_output_channels) const = 0;

    /** Gets the stride between matrices in the output worspace
     *
     * @param[in] num_batches         Number of batches in the output tensor.
     * @param[in] num_rows            Number of rows in each feature map of the input tensor.
     * @param[in] num_cols            Number of columns in each feature map of the input tensor.
     * @param[in] num_output_channels Number of feature maps in the output tensor.
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(int num_batches, int num_rows, int num_cols, int num_output_channels) const = 0;

    /** Get the output shape of a convolution.
     *
     * @param[in] num_rows     Number of rows in each feature map of the input tensor.
     * @param[in] num_cols     Number of columns in each feature map of the input tensor.
     * @param[in] padding_same True if padding is SAME, false otherwise
     *
     * @return Shape of the output tensor
     */
    virtual std::pair<unsigned int, unsigned int> get_output_shape(
        int  num_rows,    /* Number of rows in each feature map of the input tensor. */
        int  num_cols,    /* Number of columns in each feature map of the input tensor. */
        bool padding_same /* True if padding is SAME, false otherwise */
    ) const = 0;

    /** Configure the output transform kernel.
     *
     * @param[in]  biases             Pointer to the biases tensor.
     * @param[in]  transformed_output Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride      Output matrix stride, can be computed with winograd::WinogradGEMM<2, 2, 3, 3>::Convolution<float, float>::get_output_matrix_stride()
     * @param[out] output_nhwc        Pointer to a tensor in NHWC data layout ordered output tensor, in the spatial domain.
     * @param[in]  num_batches        Number of batches in the input tensor.
     * @param[in]  num_rows           Number of rows in output tensor.
     * @param[in]  num_cols           Number of columns in output tensor.
     * @param[in]  num_channels       Number of feature maps in the output tensor.
     * @param[in]  workspace          Tensor to be used as the working space during the computation.
     * @param[in]  activation         Activation to be used
     */
    virtual void configure(
        const ITensorInfo          *biases,
        const ITensorInfo          *transformed_output,
        const int                   matrix_stride,
        ITensorInfo                *output_nhwc,
        const int                   num_batches,
        const int                   num_rows,
        const int                   num_cols,
        const int                   num_channels,
        ITensorInfo                *workspace,
        const arm_gemm::Activation &activation) = 0;

    virtual ~ICpuWinogradConv2dTransformOutputKernel()
    {
    }
};

/** Kernel to perform Winograd output transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class CpuWinogradConv2dTransformOutputKernel : public ICpuWinogradConv2dTransformOutputKernel
{
public:
    const char *name() const override
    {
        return "CpuWinogradConv2dTransformOutputKernel";
    }
    /** Constructor */
    CpuWinogradConv2dTransformOutputKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformOutputKernel(const CpuWinogradConv2dTransformOutputKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformOutputKernel &operator=(const CpuWinogradConv2dTransformOutputKernel &) = delete;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformOutputKernel(CpuWinogradConv2dTransformOutputKernel &&) = default;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformOutputKernel &operator=(CpuWinogradConv2dTransformOutputKernel &&) = default;
    /** Default destructor */
    ~CpuWinogradConv2dTransformOutputKernel() = default;

    // Inherited methods overridden:
    /** Determine how much memory (in units of TOut) to allocate for the
     * (Winograd domain) output.
     *
     * @param[in] num_batches         Number of batches in the output tensor.
     * @param[in] num_rows            Number of rows in each feature map of the input tensor.
     * @param[in] num_cols            Number of columns in each feature map of the input tensor.
     * @param[in] num_output_channels Number of feature maps in the output tensor.
     *
     * @return Storage size (in units of TOut) required.
     */
    unsigned int get_output_storage_size(int num_batches, int num_rows, int num_cols, int num_output_channels) const override;

    /** Gets the stride between matrices in the output worspace
     *
     * @param[in] num_batches         Number of batches in the output tensor.
     * @param[in] num_rows            Number of rows in each feature map of the input tensor.
     * @param[in] num_cols            Number of columns in each feature map of the input tensor.
     * @param[in] num_output_channels Number of feature maps in the output tensor.
     *
     * @return Stride expressed in bytes.
     */
    int get_matrix_stride(int num_batches, int num_rows, int num_cols, int num_output_channels) const override;
    /** Get the output shape of a convolution.
     *
     * @param[in] num_rows     Number of rows in each feature map of the input tensor.
     * @param[in] num_cols     Number of columns in each feature map of the input tensor.
     * @param[in] padding_same True if padding is SAME, false otherwise
     *
     * @return Shape of the output tensor
     */
    std::pair<unsigned int, unsigned int> get_output_shape(
        int  num_rows, /* Number of rows in each feature map of the input tensor. */
        int  num_cols, /* Number of columns in each feature map of the input tensor. */
        bool padding_same) const override;

    /** Get the working space required to perform the transformation.
     *
     * Note, the working space is only required when performing the
     * transformation - hence it can be reused whenever the transformation is
     * not running.
     *
     * @param[in] num_threads The greatest number of threads that will be used to execute the transform.
     *
     * @return Size of working space required in bytes.
     */
    unsigned int get_working_space_size(unsigned int num_threads) const override;

    /** Configure the output transform kernel.
     *
     * @param[in]  biases             Pointer to the biases tensor.
     * @param[in]  transformed_output Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride      Output matrix stride, can be computed with winograd::WinogradGEMM<2, 2, 3, 3>::Convolution<float, float>::get_output_matrix_stride()
     * @param[out] output_nhwc        Pointer to a tensor with NHWC data layout, in the spatial domain.
     * @param[in]  num_batches        Number of batches in the input tensor.
     * @param[in]  num_rows           Number of rows in output tensor.
     * @param[in]  num_cols           Number of columns in output tensor.
     * @param[in]  num_channels       Number of feature maps in the output tensor.
     * @param[in]  workspace          Tensor to be used as the working space during the computation.
     * @param[in]  activation         Activation to be used
     */
    void configure(
        const ITensorInfo          *biases,
        const ITensorInfo          *transformed_output,
        const int                   matrix_stride,
        ITensorInfo                *output_nhwc,
        const int                   num_batches,
        const int                   num_rows,
        const int                   num_cols,
        const int                   num_channels,
        ITensorInfo                *workspace,
        const arm_gemm::Activation &activation) override;

    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2dTransformOutputKernel
     *
     * @param[in] input         Source tensor info with shape [C, N, 16, batches] or [C, N, 36, batches]. Data types supported: F16/F32.
     * @param[in] bias          Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. It can be a nullptr. Data type supported: as @p input
     * @param[in] output        Destination tensor info with shape [output_convolved_dims.width, output_convolved_dims.height, C, batches]. Data type supported: same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info);

private:
    using WinogradBase    = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols, winograd::WinogradRoots::Integers>;
    using WinogradConv    = typename WinogradBase::template Convolution<T, T>;
    using OutputTransform = typename WinogradBase::template OutputTransform<T, T>;

    std::unique_ptr<OutputTransform> _transform{ nullptr };
    int                              _matrix_stride;
    int                              _matrix_row_stride;
};

/** Interface for the kernel to perform Winograd weights transform. */
class ICpuWinogradConv2dTransformWeightsKernel : public NewICpuKernel<ICpuWinogradConv2dTransformWeightsKernel>
{
public:
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICpuWinogradConv2dTransformWeightsKernel(const ICpuWinogradConv2dTransformWeightsKernel &) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICpuWinogradConv2dTransformWeightsKernel &operator=(const ICpuWinogradConv2dTransformWeightsKernel &) = default;
    /** Allow instances of this class to be moved */
    ICpuWinogradConv2dTransformWeightsKernel(ICpuWinogradConv2dTransformWeightsKernel &&) = default;
    /** Allow instances of this class to be moved */
    ICpuWinogradConv2dTransformWeightsKernel &operator=(ICpuWinogradConv2dTransformWeightsKernel &&) = default;

    ICpuWinogradConv2dTransformWeightsKernel()
    {
    }
    virtual ~ICpuWinogradConv2dTransformWeightsKernel()
    {
    }
    /** Determine how much memory (in units of T) to allocate for the
     * transformed weights.
     *
     * @param[in] num_output_channels Number of output feature maps.
     * @param[in] num_input_channels  Number of input feature maps.
     *
     * @return Storage size (in units of T) required.
     */
    virtual unsigned int get_weight_storage_size(int num_output_channels, int num_input_channels) const = 0;
    /** Gets the stride between matrices in the kernel worspace
     *
     * @param[in] num_output_channels Number of output feature maps.
     * @param[in] num_input_channels  Number of input feature maps.
     *
     * @return Stride expressed in bytes.
     */
    virtual int get_matrix_stride(int num_output_channels, int num_input_channels) const = 0;

    /** Configure the weights transform kernel.
     *
     * @param[in]  weights_hwio        Pointer to the weights tensor info
     * @param[out] output              Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride       Stride across matrices in the output workspace.
     * @param[in]  num_output_channels Number of filters.
     * @param[in]  num_input_channels  Number of channels in each filter.
     */

    virtual void configure(const ITensorInfo *weights_hwio, ITensorInfo *output, const int matrix_stride, const int num_output_channels, const int num_input_channels) = 0;

    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2dTransformWeightsKernel
     *
     * @param[in] input   First tensor input info. Data types supported: F16/F32.
     * @param[in] weights Weights tensor info. Data types supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights);
};

/** Kernel to perform Winograd weights transform. */
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class CpuWinogradConv2dTransformWeightsKernel final : public ICpuWinogradConv2dTransformWeightsKernel
{
public:
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformWeightsKernel(const CpuWinogradConv2dTransformWeightsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformWeightsKernel &operator=(const CpuWinogradConv2dTransformWeightsKernel &) = delete;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformWeightsKernel(CpuWinogradConv2dTransformWeightsKernel &&) = default;
    /** Allow instances of this class to be moved */
    CpuWinogradConv2dTransformWeightsKernel &operator=(CpuWinogradConv2dTransformWeightsKernel &&) = default;
    /** Default destructor */
    ~CpuWinogradConv2dTransformWeightsKernel() = default;

    /** Default constructor. */
    CpuWinogradConv2dTransformWeightsKernel();
    const char *name() const override
    {
        return "CpuWinogradConv2dTransformWeightsKernel";
    }

    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2dTransformWeightsKernel
     *
     * @param[in] input         Source tensor info. The input is a 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] (NCHW data layout).
     *                          kernel_x must be 3 and equal to kernel_y. Data types supported: F16/F32.
     * @param[in] output        Destination tensor info. The output is a 3D tensor with dimensions [OFM, IFM, 16] or [OFM, IFM, 36]. Data type supported: same as @p input
     * @param[in] winograd_info Contains Winograd's information described in @ref WinogradInfo
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info);

    // Inherited methods overridden:

#ifndef DOXYGEN_SKIP_THIS
    /** Configure the weights transform kernel.
     *
     * @param[in]  weights_hwio        Pointer to the weights tensor info
     * @param[out] output              Pointer to working space for the output tensor in the Winograd domain.
     * @param[in]  matrix_stride       Stride across matrices in the output workspace.
     * @param[in]  num_output_channels Number of filters.
     * @param[in]  num_input_channels  Number of channels in each filter.
     */
    void configure(const ITensorInfo *weights_hwio, ITensorInfo *output, const int matrix_stride, const int num_output_channels, const int num_input_channels) override;
#endif /* DOXYGEN_SKIP_THIS */

    /** Determine how much memory (in units of T) to allocate for the
     * transformed weights.
     *
     * @param[in] num_output_channels Number of output feature maps.
     * @param[in] num_input_channels  Number of input feature maps.
     *
     * @return Storage size (in units of T) required.
     */
    unsigned int get_weight_storage_size(int num_output_channels, int num_input_channels) const override;

    /** Gets the stride between matrices in the input worspace
     *
     * @param[in] num_output_channels Number of output feature maps.
     * @param[in] num_input_channels  Number of input feature maps.
     *
     * @return Stride expressed in bytes.
     */
    int get_matrix_stride(int num_output_channels, int num_input_channels) const override;
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    using WinogradBase     = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols, winograd::WinogradRoots::Integers>;
    using WinogradConv     = typename WinogradBase::template Convolution<T, T>;
    using WeightsTransform = typename WinogradBase::template WeightsTransform<T, T>;

    std::unique_ptr<WeightsTransform> _transform{ nullptr };
    int                               _num_output_channels;
    int                               _matrix_stride;
};

/** Kernel to perform Winograd. */
template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
class CpuWinogradConv2dConfiguration
{
public:
    /** Winograd base kernel */
    using WinogradBase = winograd::WinogradGEMM<OutputTileRows, OutputTileCols, KernelRows, KernelCols, winograd::WinogradRoots::Integers>;
    /** Winograd convolution kernel */

    using WinogradConv = typename WinogradBase::template Convolution<TIn, TOut>;

    using TransformInputKernel   = CpuWinogradConv2dTransformInputKernel<TIn, OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using TransformWeightsKernel = CpuWinogradConv2dTransformWeightsKernel<TIn, OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
    using TransformOutputKernel  = CpuWinogradConv2dTransformOutputKernel<TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>;
};

} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H*/
