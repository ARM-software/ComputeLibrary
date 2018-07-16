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
#include "arm_compute/core/NEON/kernels/NEWinogradConvolutionLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
//Batched Gemms

namespace
{
Status validate_arguments_winograd_weight_trans(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);

    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_width) != 3 && input->dimension(idx_width) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_width) != input->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    const Size2D &output_tile = winograd_info.output_tile_size;
    ARM_COMPUTE_RETURN_ERROR_ON(output_tile != Size2D(2U, 2U) && output_tile != Size2D(4U, 4U));

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_filter_transform_shape(*input, winograd_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_winograd_weight_trans(ITensorInfo *input, ITensorInfo *output, const WinogradInfo &winograd_info)
{
    const Size2D kernel_dims = winograd_info.kernel_size;
    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_filter_transform_shape(*input, winograd_info)));

    unsigned int num_elems_processed_per_iteration_x = kernel_dims.width;
    unsigned int num_elems_processed_per_iteration_y = kernel_dims.height;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    bool   window_changed = false;

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    AccessWindowStatic    output_access(output, 0, 0, output->dimension(0), output->dimension(1));
    window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));

    Window win_collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};

    return std::make_pair(err, win_collapsed);
}

Status validate_arguments_winograd_input_trans(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    const Size2D        &kernel_dims = winograd_info.kernel_size;
    const PadStrideInfo &conv_info   = winograd_info.convolution_info;
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd input transform only supports unit strides");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((kernel_dims.width != 3U && kernel_dims.width != 5U), "Winograd input transform only supports 3x3 and 5x5 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((kernel_dims.width != kernel_dims.height), "Winograd input transform only supports 3x3 and 5x5 kernels");

    // Validate configured output
    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_winograd_input_trans(ITensorInfo *input, ITensorInfo *output, const WinogradInfo &winograd_info)
{
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_dims      = winograd_info.kernel_size;
    const TensorShape   output_shape     = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));

    unsigned int num_elems_read_per_iteration_x = (output_tile_size.width + kernel_dims.width - 1);
    unsigned int num_elems_read_per_iteration_y = (output_tile_size.height + kernel_dims.height - 1);

    Window win = calculate_max_window(*input, Steps(1, 1));

    AccessWindowRectangle input_access(input, -conv_info.pad_left(), -conv_info.pad_top(), num_elems_read_per_iteration_x, num_elems_read_per_iteration_y);

    bool window_changed = update_window_and_padding(win, input_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments_winograd_output_trans(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    const PadStrideInfo &conv_info   = winograd_info.convolution_info;
    const Size2D         kernel_dims = winograd_info.kernel_size;

    // Number of tiles along the X and Y direction
    const unsigned int num_tiles_x = std::ceil((winograd_info.input_dimensions.x() - (kernel_dims.width - 1) + conv_info.pad_left() + conv_info.pad_right()) / static_cast<float>
                                               (winograd_info.output_tile_size.width));
    const unsigned int num_tiles_y = std::ceil((winograd_info.input_dimensions.y() - (kernel_dims.height - 1) + conv_info.pad_top() + conv_info.pad_bottom()) / static_cast<float>
                                               (winograd_info.output_tile_size.height));
    const Size2D       num_tiles   = Size2D(num_tiles_x, num_tiles_y);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != num_tiles.area());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((kernel_dims.width != 3U && kernel_dims.width != 5U), "Winograd output transform only supports 3x3 and 5x5 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((kernel_dims.width != kernel_dims.height), "Winograd output transform only supports 3x3 and 5x5 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((input->dimension(2) != size_t(16U)) && (input->dimension(2) != size_t(36U))), "Only 2x2 and 4x4 output tile is supported");
    ARM_COMPUTE_UNUSED(kernel_dims);
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() != size_t(1));
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_output_transform_shape(*input, winograd_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_winograd_output_trans(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output, const WinogradInfo &winograd_info)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_output_transform_shape(*input, winograd_info)));

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);
    AccessWindowStatic    output_access(output, 0, 0, ceil_to_multiple(output->dimension(0), 2), ceil_to_multiple(output->dimension(1), 2));

    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
        window_changed = update_window_and_padding(win, input_access, bias_access, output_access);
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access, output_access);
    }
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

// Weights transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_weight_storage_size(int num_output_channels, int num_input_channels) const
{
    const KernelShape shape(num_output_channels, KernelRows, KernelCols, num_input_channels);
    return static_cast<unsigned int>(
               // WinogradConv returns the size in bytes, we divide by `sizeof(T)` to express that in units of T
               WinogradConv::get_kernel_storage_size(shape) / sizeof(T));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerTransformWeightsKernel()
    : _weights_hwio(nullptr), _output(nullptr), _matrix_stride(0), _num_output_channels(0), _num_input_channels(0)

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
    ITensor       *output,
    const int      matrix_stride,       /** Stride across matrices in the output. */
    const int      num_output_channels, /** Number of filters. */
    const int      num_input_channels)  /** Number of channels in each filter. */
{
    _weights_hwio        = weights_hwio;
    _output              = output;
    _matrix_stride       = matrix_stride;
    _num_output_channels = num_output_channels;
    _num_input_channels  = num_input_channels;

    const int        matrix_row_stride = roundup(num_output_channels, WinogradConv::N_BLOCK);
    WeightsTransform transform(nullptr, nullptr, matrix_stride, matrix_row_stride, num_output_channels, num_input_channels);
    Window           win;
    auto             win_last = transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    const int        matrix_row_stride = roundup(_num_output_channels, WinogradConv::N_BLOCK);
    WeightsTransform transform(reinterpret_cast<T *>(_weights_hwio->buffer()), reinterpret_cast<T *>(_output->buffer()), _matrix_stride, matrix_row_stride, _num_output_channels, _num_input_channels);
    const size_t     fst = window.x().start();
    const size_t     lst = window.x().end();
    transform.run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
bool NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::is_parallelisable() const
{
    return false;
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status NEWinogradLayerTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                                                                                  const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_weight_trans(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_weight_trans(input->clone().get(), output->clone().get(), winograd_info).first);
    return Status{};
}

template class NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformWeightsKernel<float, 4, 4, 3, 3>;
template class NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>;

// Input transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_input_storage_size(
    int  num_batches,  /* Number of batches in the input tensor. */
    int  num_channels, /* Number of feature maps in the input tensor. */
    int  num_rows,     /* Number of rows in each feature map. */
    int  num_cols,     /* Number of columns in each feature map. */
    bool same_padding  /* Use "SAME" padding, otherwise use "VALID". */
) const
{
    // Construct shapes for the input and kernel tensors.
    const Tensor4DShape input_shape(num_batches, num_rows, num_cols, num_channels);
    const KernelShape   kern_shape(1, KernelRows, KernelCols, num_channels);
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
    : _input_nhwc(), _num_batches(0), _num_rows(0), _num_cols(0), _num_channels(0), _padding(), _output(nullptr), _matrix_stride(0)
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensor    *input_nhwc,
    const int         num_batches,   /* Number of batches in input tensor. */
    const int         num_rows,      /* Number of rows in input tensor. */
    const int         num_cols,      /* Number of columns in input tensor. */
    const int         num_channels,  /* Number of channels in input tensor. */
    const PaddingType padding,       /* Padding type. */
    ITensor          *output,        /* Base of output matrices. */
    const int         matrix_stride) /* Stride between output matrices. */
{
    _input_nhwc    = input_nhwc;
    _num_batches   = num_batches;
    _num_rows      = num_rows;
    _num_cols      = num_cols;
    _num_channels  = num_channels;
    _padding       = padding;
    _output        = output;
    _matrix_stride = matrix_stride;
    InputTransform transform(nullptr, num_batches, num_rows, num_cols, num_channels, padding, nullptr, matrix_stride, num_channels);
    Window         win;
    auto           win_last = transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    InputTransform input_transform(reinterpret_cast<const T *>(_input_nhwc->buffer()), _num_batches, _num_rows, _num_cols, _num_channels, _padding, reinterpret_cast<T *>(_output->buffer()),
                                   _matrix_stride, _num_channels);

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    input_transform.run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status NEWinogradLayerTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_input_trans(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_input_trans(input->clone().get(), output->clone().get(), winograd_info).first);

    return Status{};
}

template class NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformInputKernel<float, 4, 4, 3, 3>;
template class NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>;

// Output transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_storage_size(
    int  num_batches,         /* Number of batches in the output tensor. */
    int  num_rows,            /* Number of rows in each feature map of the input tensor. */
    int  num_cols,            /* Number of columns in each feature map of the input tensor. */
    int  num_output_channels, /* Number of feature maps in the output tensor. */
    bool same_padding         /* Use "SAME" padding, otherwise use "VALID". */
) const
{
    // Construct shapes for the input and kernel tensors.
    const Tensor4DShape input_shape(num_batches, num_rows, num_cols, 1);
    const KernelShape   kern_shape(num_output_channels, KernelRows, KernelCols, 1);
    const PaddingType   padding = (same_padding) ? PADDING_SAME : PADDING_VALID;

    // Return the size, converted into units of TOut
    return static_cast<unsigned int>(
               WinogradConv::get_output_storage_size(kern_shape, input_shape, padding) / sizeof(T));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::NEWinogradLayerTransformOutputKernel()
    : _biases(nullptr), _output_workspace(nullptr), _matrix_stride(0), _matrix_row_stride(0), _output_nhwc(nullptr), _num_batches(0), _num_rows(0), _num_cols(0), _num_channels(0)
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
    const ITensor *output_workingspace,
    const int      matrix_stride,
    ITensor       *output_nhwc,
    const int      num_batches,
    const int      num_rows,
    const int      num_cols,
    const int      num_channels)
{
    _biases            = biases;
    _output_workspace  = output_workingspace;
    _matrix_stride     = matrix_stride;
    _matrix_row_stride = roundup(num_channels, WinogradConv::N_BLOCK);
    _output_nhwc       = output_nhwc;
    _num_batches       = num_batches;
    _num_rows          = num_rows;
    _num_cols          = num_cols;
    _num_channels      = num_channels;
    // We don't have the biases buffer at this stage as it hasn't been allocated, we pass in nullptr OutputTransform is only used here to compute the window
    OutputTransform output_transform(nullptr, _matrix_stride, _matrix_row_stride, nullptr, nullptr, _num_batches, _num_rows, _num_cols, _num_channels);

    Window win;
    auto   win_last = output_transform.get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));

    _output_nhwc->info()->set_valid_region(ValidRegion(Coordinates(), _output_nhwc->info()->tensor_shape()));

    INEKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output_workspace);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_output_nhwc);

    OutputTransform output_transform(reinterpret_cast<T *>(_output_workspace->buffer()), _matrix_stride, _matrix_row_stride,
                                     (_biases ? reinterpret_cast<T *>(_biases->buffer()) : nullptr), reinterpret_cast<T *>(_output_nhwc->buffer()),
                                     _num_batches, _num_rows, _num_cols, _num_channels, 0, _output_nhwc->info()->strides_in_bytes()[2] / sizeof(T), _output_nhwc->info()->strides_in_bytes()[1] / sizeof(T));

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    output_transform.run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status NEWinogradLayerTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                                                                 const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_output_trans(input, (bias != nullptr ? bias->clone().get() : nullptr), output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_output_trans(input->clone().get(), (bias != nullptr ? bias->clone().get() : nullptr), output->clone().get(),
                                                                                    winograd_info)
                                .first);

    return Status{};
}

template class NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>;
template class NEWinogradLayerTransformOutputKernel<float, 4, 4, 3, 3>;
template class NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>;

} // namespace arm_compute
