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

#include "arm_compute/core/NEON/kernels/winograd/winograd_layer.hpp"

namespace
{
using T = WinogradConvolutionLayer<2, 2, 3, 3, float, float>;
} // namespace

namespace arm_compute
{
class Winograd3x3F32::Private
{
public:
    Private(
        const int          n_batches,         /** Number of batches in the input and output tensors. */
        const int          n_input_channels,  /** Number of feature maps in a batch of the input tensor. */
        const int          n_input_rows,      /** Number of rows in a feature map of the input tensor. */
        const int          n_input_cols,      /** Number of columns in a feature map of the input tensor. */
        const int          n_output_channels, /** Number of feature maps in the output tensor. */
        const bool         same_padding,      /** Use "SAME" padding, otherwise use "VALID". */
        const float *const weights,           /** Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps. */
        float *const       weights_storage,   /** Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size`. */
        const float *const input,             /** Pointer to NHWC ordered input tensor, in the spatial domain. */
        float *const       winograd_input,    /** Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`. */
        float *const       output,            /** Pointer to NHWC ordered output tensor, in the spatial domain. */
        float *const       winograd_output    /** Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`. */
    )
        : convolver(n_batches, n_input_channels, n_input_rows, n_input_cols, n_output_channels, same_padding, weights, weights_storage, input, winograd_input, nullptr, output, winograd_output)
    {
    }
    T convolver;
};

Winograd3x3F32::~Winograd3x3F32()
{
}

Winograd3x3F32::Winograd3x3F32(
    const int          n_batches,         /** Number of batches in the input and output tensors. */
    const int          n_input_channels,  /** Number of feature maps in a batch of the input tensor. */
    const int          n_input_rows,      /** Number of rows in a feature map of the input tensor. */
    const int          n_input_cols,      /** Number of columns in a feature map of the input tensor. */
    const int          n_output_channels, /** Number of feature maps in the output tensor. */
    const bool         same_padding,      /** Use "SAME" padding, otherwise use "VALID". */
    const float *const weights,           /** Pointer to weight tensor in spatial domain. Must be ordered as "Height x Rows x Input Feature Maps x Output Feature Maps. */
    float *const       weights_storage,   /** Pointer to storage for weight tensor in the Winograd domain. Must be at least the size returned by `get_weight_storage_size`. */
    const float *const input,             /** Pointer to NHWC ordered input tensor, in the spatial domain. */
    float *const       winograd_input,    /** Pointer to working space for the input tensor in the Winograd domain. Must be at least the size returned by `get_input_storage_size`. */
    float *const       output,            /** Pointer to NHWC ordered output tensor, in the spatial domain. */
    float *const       winograd_output    /** Pointer to working space for the output tensor in the Winograd domain. Must be at least the size returned by `get_output_storage_size`. */
)
    : _pimpl(support::cpp14::make_unique<Private>(n_batches, n_input_channels, n_input_rows, n_input_cols, n_output_channels, same_padding, weights, weights_storage, input, winograd_input, output,
                                                  winograd_output))
{
}

unsigned int NEWinogradLayerKernel::get_input_storage_size(const int n_batches, const int n_channels, const int n_rows, const int n_cols, const bool same_padding)
{
    return T::get_input_storage_size(n_batches, n_channels, n_rows, n_cols, same_padding);
}

unsigned int NEWinogradLayerKernel::get_output_storage_size(
    const int  n_batches,         /** Number of batches in the output tensor. */
    const int  n_rows,            /** Number of rows in each feature map of the input tensor. */
    const int  n_cols,            /** Number of columns in each feature map of the input tensor. */
    const int  n_output_channels, /** Number of feature maps in the output tensor. */
    const bool same_padding       /** Use "SAME" padding, otherwise use "VALID". */
)
{
    return T::get_output_storage_size(n_batches, n_rows, n_cols, n_output_channels, same_padding);
}

unsigned int NEWinogradLayerKernel::get_weight_storage_size(const int n_output_channels, const int n_input_channels)
{
    return T::get_weight_storage_size(n_output_channels, n_input_channels);
}

NEWinogradLayerKernel::NEWinogradLayerKernel()
    : _convolver(nullptr)
{
}

void NEWinogradLayerKernel::configure(Winograd3x3F32 *convolver)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(convolver);
    _convolver = convolver;
    Window win;
    auto   win_last = _convolver->_pimpl->convolver.gemms.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

void NEWinogradLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t first_gemm = window.x().start();
    const size_t last_gemm  = window.x().end();
    _convolver->_pimpl->convolver.gemms.run(first_gemm, last_gemm);
}

INEWinogradLayerTransformKernel::INEWinogradLayerTransformKernel()
    : _convolver(nullptr)
{
}

void INEWinogradLayerTransformKernel::configure(Winograd3x3F32 *convolver)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(convolver);
    _convolver = convolver;
}

// Weights transform

void NEWinogradLayerTransformWeightsKernel::configure(Winograd3x3F32 *convolver)
{
    INEWinogradLayerTransformKernel::configure(convolver);
    Window win;
    auto   win_last = _convolver->_pimpl->convolver.weights_transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

void NEWinogradLayerTransformWeightsKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _convolver->_pimpl->convolver.weights_transform.run(fst, lst);
}

bool NEWinogradLayerTransformWeightsKernel::is_parallelisable() const
{
    return false;
}

// Input transform

void NEWinogradLayerTransformInputKernel::configure(Winograd3x3F32 *convolver)
{
    INEWinogradLayerTransformKernel::configure(convolver);
    Window win;
    auto   win_last = _convolver->_pimpl->convolver.input_transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

void NEWinogradLayerTransformInputKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _convolver->_pimpl->convolver.input_transform.run(fst, lst);
}
bool NEWinogradLayerTransformInputKernel::is_parallelisable() const
{
    return false;
}

// Output transform
NEWinogradLayerTransformOutputKernel::NEWinogradLayerTransformOutputKernel()
    : _biases(nullptr), _output_workspace(nullptr), _matrix_stride(0), _matrix_row_stride(0), _output(nullptr), _n_batches(0), _n_rows(0), _n_cols(0), _n_channels(0)
{
}

void NEWinogradLayerTransformOutputKernel::configure(
    const ITensor     *biases,
    const float *const output_workingspace,
    const int          matrix_stride,
    float *const       output,
    const int          n_batches,
    const int          n_rows,
    const int          n_cols,
    const int          n_channels)
{
    using WinogradBase    = winograd::WinogradGEMM<2, 2, 3, 3>;
    using OutputTransform = typename WinogradBase::template OutputTransform<float>;

    _biases            = biases;
    _output_workspace  = output_workingspace;
    _matrix_stride     = matrix_stride;
    _matrix_row_stride = roundup(n_channels, WinogradBase::Convolution<float, float>::N_BLOCK);
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

void NEWinogradLayerTransformOutputKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_biases->buffer());
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output_workspace);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output);

    using WinogradBase    = winograd::WinogradGEMM<2, 2, 3, 3>;
    using OutputTransform = typename WinogradBase::template OutputTransform<float>;

    OutputTransform output_transform(_output_workspace, _matrix_stride, _matrix_row_stride,
                                     reinterpret_cast<float *>(_biases->buffer()), _output,
                                     _n_batches, _n_rows, _n_cols, _n_channels);

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    output_transform.run(fst, lst);
}

bool NEWinogradLayerTransformOutputKernel::is_parallelisable() const
{
    return false;
}

} // namespace arm_compute
