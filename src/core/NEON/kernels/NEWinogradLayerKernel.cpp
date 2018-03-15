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
#include "arm_compute/core/NEON/kernels/NEWinogradLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
//Batched Gemms
template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerBatchedGEMMKernel()
    : _gemms()
{
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
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
    TOut *const      c_ptr)
{
    _gemms = support::cpp14::make_unique<MultiGEMM>(n_gemms, M, K, N, a_matrix_stride, a_row_stride, b_matrix_stride, b_row_stride, c_matrix_stride, c_row_stride, a_ptr, b_ptr, c_ptr);
    Window win;
    auto   win_last = _gemms->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t first_gemm = window.x().start();
    const size_t last_gemm  = window.x().end();
    _gemms->run(first_gemm, last_gemm);
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_number_gemms() const
{
    return WinogradBase::N_GEMMS;
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_tile_rows() const
{
    return _output_tile_rows;
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_tile_cols() const
{
    return _output_tile_cols;
}

template <typename TIn, typename TOut, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerBatchedGEMMKernel<TIn, TOut, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_number_blocks() const
{
    return WinogradConv::N_BLOCK;
}

template class NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 3, 3>;
template class NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 5, 5>;

// Weights transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_weight_storage_size(int n_output_channels, int n_input_channels) const
{
    const KernelShape shape(n_output_channels, KernelRows, KernelCols, n_input_channels);
    return static_cast<unsigned int>(
               // WinogradConv returns the size in bytes, we divide by `sizeof(T)` to express that in units of T
               WinogradConv::get_kernel_storage_size(shape) / sizeof(T));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerTransformWeightsKernel()
    : _transform()
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(const KernelShape &kernel_shape) const
{
    return WinogradConv::get_kernel_matrix_stride(kernel_shape);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensor *weights_hwio,
    T *const       output,
    const int      matrix_stride,     /** Stride across matrices in the output. */
    const int      n_output_channels, /** Number of filters. */
    const int      n_input_channels)  /** Number of channels in each filter. */
{
    const int matrix_row_stride = roundup(n_output_channels, WinogradConv::N_BLOCK);
    _transform                  = support::cpp14::make_unique<WeightsTransform>(reinterpret_cast<T *>(weights_hwio->buffer()), output, matrix_stride, matrix_row_stride, n_output_channels,
                                                                                n_input_channels);
    Window win;
    auto   win_last = _transform->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _transform->run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
bool NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::is_parallelisable() const
{
    return false;
}

template class NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>;

// Input transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_input_storage_size(
    int  n_batches,   /** Number of batches in the input tensor. */
    int  n_channels,  /** Number of feature maps in the input tensor. */
    int  n_rows,      /** Number of rows in each feature map. */
    int  n_cols,      /** Number of columns in each feature map. */
    bool same_padding /** Use "SAME" padding, otherwise use "VALID". */
) const
{
    // Construct shapes for the input and kernel tensors.
    const Tensor4DShape input_shape(n_batches, n_rows, n_cols, n_channels);
    const KernelShape   kern_shape(1, KernelRows, KernelCols, n_channels);
    const PaddingType   padding = (same_padding) ? PADDING_SAME : PADDING_VALID;
    // Return the size, converted into units of TIn
    return static_cast<unsigned int>(WinogradConv::get_input_storage_size(kern_shape, input_shape, padding) / sizeof(T));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(
    const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const
{
    return WinogradConv::get_input_matrix_stride(kernel_shape, input_shape, padding_type);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerTransformInputKernel()
    : _transform()
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const T *const    input,         /** Input tensor data */
    const int         n_batches,     /** Number of batches in input tensor. */
    const int         n_rows,        /** Number of rows in input tensor. */
    const int         n_cols,        /** Number of columns in input tensor. */
    const int         n_channels,    /** Number of channels in input tensor. */
    const PaddingType padding,       /** Padding type. */
    T *const          output,        /** Base of output matrices. */
    const int         matrix_stride) /** Stride between output matrices. */
{
    //  _input_matrix_row_stride(n_input_channels),
    _transform = support::cpp14::make_unique<InputTransform>(input, n_batches, n_rows, n_cols, n_channels, padding, output, matrix_stride, n_channels);
    Window win;
    auto   win_last = _transform->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _transform->run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
bool NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::is_parallelisable() const
{
    return false;
}

template class NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>;

// Output transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_storage_size(
    int  n_batches,         /** Number of batches in the output tensor. */
    int  n_rows,            /** Number of rows in each feature map of the input tensor. */
    int  n_cols,            /** Number of columns in each feature map of the input tensor. */
    int  n_output_channels, /** Number of feature maps in the output tensor. */
    bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
) const
{
    // Construct shapes for the input and kernel tensors.
    const Tensor4DShape input_shape(n_batches, n_rows, n_cols, 1);
    const KernelShape   kern_shape(n_output_channels, KernelRows, KernelCols, 1);
    const PaddingType   padding = (same_padding) ? PADDING_SAME : PADDING_VALID;

    // Return the size, converted into units of TOut
    return static_cast<unsigned int>(
               WinogradConv::get_output_storage_size(kern_shape, input_shape, padding) / sizeof(T));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerTransformOutputKernel()
    : _biases(nullptr), _output_workspace(nullptr), _matrix_stride(0), _matrix_row_stride(0), _output(nullptr), _n_batches(0), _n_rows(0), _n_cols(0), _n_channels(0)
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(
    const KernelShape &kernel_shape, const Tensor4DShape &input_shape, const PaddingType padding_type) const
{
    return WinogradConv::get_output_matrix_stride(kernel_shape, input_shape, padding_type);
}
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Tensor4DShape NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_shape(
    const KernelShape &kernel_shape, const Tensor4DShape &in_shape, const PaddingType padding) const
{
    return WinogradConv::get_output_shape(kernel_shape, in_shape, padding);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensor *biases,
    const T *const output_workingspace,
    const int      matrix_stride,
    T *const       output,
    const int      n_batches,
    const int      n_rows,
    const int      n_cols,
    const int      n_channels)
{
    _biases            = biases;
    _output_workspace  = output_workingspace;
    _matrix_stride     = matrix_stride;
    _matrix_row_stride = roundup(n_channels, WinogradConv::N_BLOCK);
    _output            = output;
    _n_batches         = n_batches;
    _n_rows            = n_rows;
    _n_cols            = n_cols;
    _n_channels        = n_channels;

    // We don't have the biases buffer at this stage as it hasn't been allocated, we pass in nullptr OutputTransform is only used here to compute the window
    OutputTransform output_transform(_output_workspace, _matrix_stride, _matrix_row_stride, nullptr, _output, _n_batches, _n_rows, _n_cols, _n_channels);
    Window          win;
    auto            win_last = output_transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output_workspace);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output);

    OutputTransform output_transform(_output_workspace, _matrix_stride, _matrix_row_stride,
                                     (_biases ? reinterpret_cast<T *>(_biases->buffer()) : nullptr), _output,
                                     _n_batches, _n_rows, _n_cols, _n_channels);

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    output_transform.run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
bool NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::is_parallelisable() const
{
    return false;
}

template class NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>;

} // namespace arm_compute
