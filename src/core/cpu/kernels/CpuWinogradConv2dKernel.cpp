/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuWinogradConv2dKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/NEON/kernels/convolution/common/utils.hpp"
#include "src/core/NEON/kernels/convolution/winograd/winograd_layer.hpp"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
//Batched Gemms

namespace
{
inline bool is_kernel_size_supported(DataType data_type, Size2D size)
{
    const std::array<Size2D, 8> f32_support = { { Size2D(1, 3), Size2D(3, 1), Size2D(5, 5), Size2D(3, 3), Size2D(1, 5), Size2D(5, 1), Size2D(7, 1), Size2D(1, 7) } };
    const std::array<Size2D, 8> f16_support = { { Size2D(3, 3) } };

    switch(data_type)
    {
        case DataType::F16:
            return std::end(f16_support) != std::find(std::begin(f16_support), std::end(f16_support), size);
        case DataType::F32:
            return std::end(f32_support) != std::find(std::begin(f32_support), std::end(f32_support), size);
        default:
            return false;
    }
}

Status validate_arguments_winograd_weight_trans(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    const size_t idx_width    = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height   = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const auto   input_width  = input->dimension(idx_width);
    const auto   input_height = input->dimension(idx_height);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_kernel_size_supported(input->data_type(), Size2D(input_width, input_height)),
                                    "Only 1x3, 3x1, 1x5, 5x1, 7x1, 1x7, 3x3 and 5x5 kernels are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    const Size2D &output_tile = winograd_info.output_tile_size;
    const std::array<Size2D, 8> supported_tile_sizes = { { Size2D(2U, 2U), Size2D(4U, 4U), Size2D(1U, 6U), Size2D(6U, 1U), Size2D(4, 1), Size2D(1, 4), Size2D(2, 1), Size2D(1, 2) } };
    ARM_COMPUTE_RETURN_ERROR_ON(std::end(supported_tile_sizes) == std::find(std::begin(supported_tile_sizes), std::end(supported_tile_sizes), output_tile));

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
    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_filter_transform_shape(*input, winograd_info)));
    const Window win = calculate_max_window(*input, Steps(), true /* skip border*/);
    return std::make_pair(Status{}, win);
}

Status validate_arguments_winograd_input_trans(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    const Size2D        &kernel_dims = winograd_info.kernel_size;
    const PadStrideInfo &conv_info   = winograd_info.convolution_info;
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd input transform only supports unit strides");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_kernel_size_supported(input->data_type(), Size2D(kernel_dims.width, kernel_dims.height)),
                                    "Only 1x3, 3x1, 3x3 and 5x5 kernels are supported");

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
    const TensorShape output_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));
    return std::make_pair(Status{}, calculate_max_window(*input, Steps(), true));
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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != num_tiles.area());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_kernel_size_supported(input->data_type(), Size2D(kernel_dims.width, kernel_dims.height)),
                                    "Only 1x3, 3x1, 3x3 and 5x5 kernels are supported");

    const std::array<unsigned int, 3> supported_gemm_sizes = { { 8U, 16U, 36U } };
    ARM_COMPUTE_RETURN_ERROR_ON(std::end(supported_gemm_sizes) == std::find(std::begin(supported_gemm_sizes), std::end(supported_gemm_sizes), input->dimension(2)));
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

std::pair<Status, Window> validate_and_configure_window_winograd_output_trans(ITensorInfo *input, ITensorInfo *output, const WinogradInfo &winograd_info)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(arm_compute::misc::shape_calculator::compute_winograd_output_transform_shape(*input, winograd_info)));

    return std::make_pair(Status{}, calculate_max_window(*input, Steps(), true));
}
} // namespace

Status ICpuWinogradConv2dTransformWeightsKernel::validate(const ITensorInfo *input, const ITensorInfo *weights)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    const DataLayout   data_layout = input->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_kernel_size_supported(input->data_type(), Size2D(weights->dimension(width_idx), weights->dimension(height_idx))),
                                    "Only 1x3, 3x1, 3x3 and 5x5 kernels are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    return Status{};
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_weight_storage_size(int num_output_channels, int num_input_channels) const
{
    const KernelShape shape(num_output_channels, KernelRows, KernelCols, num_input_channels);
    return static_cast<unsigned int>(
               WinogradConv::get_kernel_storage_size(num_input_channels, num_output_channels));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::CpuWinogradConv2dTransformWeightsKernel()
    : _transform(nullptr), _num_output_channels(0), _matrix_stride(0)
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(int num_output_channels, int num_input_channels) const
{
    return WinogradConv::get_kernel_matrix_stride(num_input_channels, num_output_channels);
}

#ifndef DOXYGEN_SKIP_THIS
template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensorInfo *weights_hwio,
    ITensorInfo       *output,
    const int          matrix_stride,       /** Stride across matrices in the output. */
    const int          num_output_channels, /** Number of filters. */
    const int          num_input_channels)  /** Number of channels in each filter. */
{
    ARM_COMPUTE_UNUSED(weights_hwio, output);

    _transform           = std::make_unique<WeightsTransform>(num_output_channels, num_input_channels);
    _num_output_channels = num_output_channels;
    _matrix_stride       = matrix_stride;

    Window win;
    auto   win_last = _transform->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    ICpuKernel::configure(win);
}
#endif /* DOXYGEN_SKIP_THIS */

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const size_t fst = window.x().start();
    const size_t lst = window.x().end();

    const ITensor *weights_hwio = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *output       = tensors.get_tensor(TensorType::ACL_DST);

    _transform->set_weight_tensor(weights_hwio->buffer());
    const int matrix_row_stride = roundup(_num_output_channels, WinogradConv::N_BLOCK);
    _transform->set_output_matrices(output->buffer(), _matrix_stride, matrix_row_stride);
    _transform->set_working_space(output->buffer());

    _transform->run(fst, lst);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
bool CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::is_parallelisable() const
{
    return false;
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status CpuWinogradConv2dTransformWeightsKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                                                                                    const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_weight_trans(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_weight_trans(input->clone().get(), output->clone().get(), winograd_info).first);
    return Status{};
}

template class CpuWinogradConv2dTransformWeightsKernel<float, 2, 2, 3, 3>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 4, 4, 3, 3>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 2, 2, 5, 5>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 1, 6, 1, 3>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 6, 1, 3, 1>;

template class CpuWinogradConv2dTransformWeightsKernel<float, 1, 4, 1, 5>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 4, 1, 5, 1>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 1, 2, 1, 7>;
template class CpuWinogradConv2dTransformWeightsKernel<float, 2, 1, 7, 1>;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class CpuWinogradConv2dTransformWeightsKernel<__fp16, 4, 4, 3, 3>;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// Input transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_input_storage_size(
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
    return static_cast<unsigned int>(WinogradConv::get_input_storage_size(num_batches, num_rows, num_cols, num_channels, same_padding));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_working_space_size(unsigned int num_threads) const
{
    return _transform->get_working_space_size(num_threads);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(
    int  num_batches,  /* Number of batches in the input tensor. */
    int  num_channels, /* Number of feature maps in the input tensor. */
    int  num_rows,     /* Number of rows in each feature map. */
    int  num_cols,     /* Number of columns in each feature map. */
    bool same_padding /* Use "SAME" padding, otherwise use "VALID". */) const
{
    return WinogradConv::get_input_matrix_stride(num_batches, num_rows, num_cols, num_channels, same_padding);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::CpuWinogradConv2dTransformInputKernel()
    : _transform(nullptr), _num_channels(0), _matrix_stride(0)
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensorInfo *input_nhwc,
    const int          num_batches,   /* Number of batches in input tensor. */
    const int          num_rows,      /* Number of rows in input tensor. */
    const int          num_cols,      /* Number of columns in input tensor. */
    const int          num_channels,  /* Number of channels in input tensor. */
    const PaddingType  padding,       /* Padding type. */
    ITensorInfo       *output,        /* Base of output matrices. */
    const int          matrix_stride, /* Stride between output matrices. */
    ITensorInfo       *workspace)
{
    ARM_COMPUTE_UNUSED(input_nhwc, output, matrix_stride, workspace);

    _num_channels  = num_channels;
    _matrix_stride = matrix_stride;

    const int padding_top    = (padding == PADDING_SAME) ? (KernelRows - 1) / 2 : 0;
    const int padding_left   = (padding == PADDING_SAME) ? (KernelCols - 1) / 2 : 0;
    const int padding_bottom = (padding == PADDING_SAME) ? iceildiv(KernelRows - 1, 2) : 0;
    const int padding_right  = (padding == PADDING_SAME) ? iceildiv(KernelCols - 1, 2) : 0;

    _transform = std::make_unique<InputTransform>(
                     KernelRows,
                     KernelCols,
                     num_batches,
                     num_rows,
                     num_cols,
                     num_channels,
                     padding_top,    /**< Padding to apply to the top of the image. */
                     padding_left,   /**< Padding to apply to the left of the image. */
                     padding_bottom, /**< Padding to apply to the bottom of the image. */
                     padding_right   /**< Padding to apply to the right of the image. */
                 );

    Window win;
    auto   win_last = _transform->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));
    ICpuKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *input_nhwc = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *workspace  = tensors.get_const_tensor(TensorType::ACL_INT);
    ITensor       *output     = tensors.get_tensor(TensorType::ACL_DST);

    const int  element_size_in_bytes = input_nhwc->info()->element_size();
    const int  input_col_stride      = input_nhwc->info()->strides_in_bytes().y() / element_size_in_bytes;
    const int  input_row_stride      = input_nhwc->info()->strides_in_bytes().z() / element_size_in_bytes;
    const int  input_batch_stride    = input_nhwc->info()->strides_in_bytes()[3] / element_size_in_bytes;
    const auto input_nhwc_ptr        = reinterpret_cast<const T *>(input_nhwc->buffer() + input_nhwc->info()->offset_first_element_in_bytes());
    auto       output_ptr            = reinterpret_cast<T *>(output->buffer() + output->info()->offset_first_element_in_bytes());
    ARM_COMPUTE_ERROR_ON_NULLPTR(output_ptr);

    _transform->set_input_tensor(input_nhwc_ptr, input_batch_stride, input_row_stride, input_col_stride);
    _transform->set_output_matrices(output_ptr, _matrix_stride, _num_channels);

    _transform->set_working_space(workspace->buffer());

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _transform->run(fst, lst, info.thread_id);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status CpuWinogradConv2dTransformInputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                                                                                  const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_input_trans(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_input_trans(input->clone().get(), output->clone().get(), winograd_info).first);

    return Status{};
}

template class CpuWinogradConv2dTransformInputKernel<float, 2, 2, 3, 3>;
template class CpuWinogradConv2dTransformInputKernel<float, 4, 4, 3, 3>;
template class CpuWinogradConv2dTransformInputKernel<float, 2, 2, 5, 5>;
template class CpuWinogradConv2dTransformInputKernel<float, 1, 6, 1, 3>;
template class CpuWinogradConv2dTransformInputKernel<float, 6, 1, 3, 1>;

template class CpuWinogradConv2dTransformInputKernel<float, 1, 4, 1, 5>;
template class CpuWinogradConv2dTransformInputKernel<float, 4, 1, 5, 1>;
template class CpuWinogradConv2dTransformInputKernel<float, 1, 2, 1, 7>;
template class CpuWinogradConv2dTransformInputKernel<float, 2, 1, 7, 1>;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class CpuWinogradConv2dTransformInputKernel<__fp16, 4, 4, 3, 3>;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// Output transform

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_storage_size(
    int num_batches,        /* Number of batches in the output tensor. */
    int num_rows,           /* Number of rows in each feature map of the input tensor. */
    int num_cols,           /* Number of columns in each feature map of the input tensor. */
    int num_output_channels /* Number of feature maps in the output tensor. */
) const
{
    // Construct shapes for the input and kernel tensors.
    const Tensor4DShape input_shape(num_batches, num_rows, num_cols, 1);
    const KernelShape   kern_shape(num_output_channels, KernelRows, KernelCols, 1);
    return static_cast<unsigned int>(
               WinogradConv::get_output_storage_size(num_batches, num_rows, num_cols, num_output_channels));
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::CpuWinogradConv2dTransformOutputKernel()
    : _transform(nullptr), _matrix_stride(0), _matrix_row_stride(0)
{
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
unsigned int CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_working_space_size(unsigned int num_threads) const
{
    return _transform->get_working_space_size(num_threads);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
int CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_matrix_stride(
    int num_batches,        /* Number of batches in the output tensor. */
    int num_rows,           /* Number of rows in each feature map of the input tensor. */
    int num_cols,           /* Number of columns in each feature map of the input tensor. */
    int num_output_channels /* Number of feature maps in the output tensor. */
) const
{
    return WinogradConv::get_output_matrix_stride(num_batches, num_rows, num_cols, num_output_channels);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
std::pair<unsigned int, unsigned int> CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::get_output_shape(
    int  num_rows, /* Number of rows in each feature map of the input tensor. */
    int  num_cols, /* Number of columns in each feature map of the input tensor. */
    bool padding_same) const
{
    return WinogradConv::get_output_shape(std::make_pair<unsigned int, unsigned int>(num_rows, num_cols), padding_same);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::configure(
    const ITensorInfo          *biases,
    const ITensorInfo          *transformed_output,
    const int                   matrix_stride,
    ITensorInfo                *output_nhwc,
    const int                   num_batches,
    const int                   num_rows,
    const int                   num_cols,
    const int                   num_channels,
    ITensorInfo                *workspace,
    const arm_gemm::Activation &activation)
{
    ARM_COMPUTE_UNUSED(biases, transformed_output, output_nhwc, num_batches, num_rows, num_cols, workspace, activation);

    _matrix_stride     = matrix_stride;
    _matrix_row_stride = roundup(num_channels, WinogradConv::N_BLOCK);

    // We don't have the biases buffer at this stage as it hasn't been allocated, we pass in nullptr OutputTransform is only used here to compute the window
    _transform = std::make_unique<OutputTransform>(num_batches, num_rows, num_cols, num_channels, activation);
    Window win;
    auto   win_last = _transform->get_window();
    win.set(Window::DimX, Window::Dimension(0, win_last, 1));

    ICpuKernel::configure(win);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
void CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *biases             = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *transformed_output = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *workspace          = tensors.get_tensor(TensorType::ACL_INT);
    ITensor       *dst_nhwc           = tensors.get_tensor(TensorType::ACL_DST);

    const int out_batch_stride = dst_nhwc->info()->strides_in_bytes()[3] / sizeof(T);
    const int out_row_stride   = dst_nhwc->info()->strides_in_bytes()[2] / sizeof(T);
    const int out_col_stride   = dst_nhwc->info()->strides_in_bytes()[1] / sizeof(T);

    _transform->set_input_matrices(transformed_output->buffer(), _matrix_stride, _matrix_row_stride);
    _transform->set_bias((biases ? reinterpret_cast<T *>(biases->buffer() + biases->info()->offset_first_element_in_bytes()) : nullptr));
    _transform->set_output_tensor(dst_nhwc->buffer() + dst_nhwc->info()->offset_first_element_in_bytes(), out_batch_stride, out_row_stride, out_col_stride);
    _transform->set_working_space(workspace->buffer());

    // The code below cannot be moved to configure because biases hasn't been allocated at that point
    const size_t fst = window.x().start();
    const size_t lst = window.x().end();
    _transform->run(fst, lst, info.thread_id);
}

template <typename T, int OutputTileRows, int OutputTileCols, int KernelRows, int KernelCols>
Status CpuWinogradConv2dTransformOutputKernel<T, OutputTileRows, OutputTileCols, KernelRows, KernelCols>::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                                                                   const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_winograd_output_trans(input, (bias != nullptr ? bias->clone().get() : nullptr), output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_winograd_output_trans(input->clone().get(), output->clone().get(), winograd_info).first);

    return Status{};
}

template class CpuWinogradConv2dTransformOutputKernel<float, 2, 2, 3, 3>;
template class CpuWinogradConv2dTransformOutputKernel<float, 4, 4, 3, 3>;
template class CpuWinogradConv2dTransformOutputKernel<float, 2, 2, 5, 5>;
template class CpuWinogradConv2dTransformOutputKernel<float, 1, 6, 1, 3>;
template class CpuWinogradConv2dTransformOutputKernel<float, 6, 1, 3, 1>;

template class CpuWinogradConv2dTransformOutputKernel<float, 1, 4, 1, 5>;
template class CpuWinogradConv2dTransformOutputKernel<float, 4, 1, 5, 1>;
template class CpuWinogradConv2dTransformOutputKernel<float, 1, 2, 1, 7>;
template class CpuWinogradConv2dTransformOutputKernel<float, 2, 1, 7, 1>;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class CpuWinogradConv2dTransformOutputKernel<__fp16, 4, 4, 3, 3>;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
} // namespace cpu
} // namespace arm_compute
