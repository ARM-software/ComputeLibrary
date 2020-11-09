/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEROIAlignLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, rois, output);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->dimension(0) != 5);
    ARM_COMPUTE_RETURN_ERROR_ON(rois->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC, DataLayout::NCHW);
    ARM_COMPUTE_RETURN_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(compute_roi_align_shape(*input, *rois, pool_info), output->tensor_shape());
    }

    if(input->data_type() == DataType::QASYMM8 || input->data_type() == DataType::QASYMM8_SIGNED)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rois, 1, DataType::QASYMM16);

        const UniformQuantizationInfo rois_qinfo = rois->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(rois_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(rois_qinfo.offset != 0);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, rois);
    }

    return Status{};
}
} // namespace

NEROIAlignLayerKernel::NEROIAlignLayerKernel()
    : _input(nullptr), _output(nullptr), _rois(nullptr), _pool_info(0, 0, 0.f)
{
}

void NEROIAlignLayerKernel::configure(const ITensor *input, const ITensor *rois, ITensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, rois);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), rois->info(), output->info(), pool_info));
    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = compute_roi_align_shape(*input->info(), *rois->info(), pool_info);
    auto_init_if_empty((*output->info()), output_shape, 1, input->info()->data_type(), input->info()->quantization_info());
    output->info()->set_data_layout(input->info()->data_layout());

    // Configure kernel window
    const unsigned int num_rois = rois->info()->dimension(1);
    Window             window;
    window.set(Window::DimX, Window::Dimension(0, num_rois));
    window.set(Window::DimY, Window::Dimension(0, 1));

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    INEKernel::configure(window);
}

Status NEROIAlignLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, rois, output, pool_info));
    return Status{};
}

/** Average pooling over an aligned window */
template <typename input_data_type>
inline input_data_type roi_align_1x1(const ITensor *input,
                                     unsigned int   roi_batch,
                                     float          region_start_x,
                                     float          bin_size_x,
                                     int            grid_size_x,
                                     float          region_end_x,
                                     float          region_start_y,
                                     float          bin_size_y,
                                     int            grid_size_y,
                                     float          region_end_y,
                                     int            pz)
{
    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
    {
        return input_data_type(0);
    }
    else
    {
        const DataLayout data_layout = input->info()->data_layout();
        float            avg         = 0;
        // Iterate through the aligned pooling region
        for(int iy = 0; iy < grid_size_y; ++iy)
        {
            for(int ix = 0; ix < grid_size_x; ++ix)
            {
                // Align the window in the middle of every bin
                float y = region_start_y + (iy + 0.5) * bin_size_y / float(grid_size_y);
                float x = region_start_x + (ix + 0.5) * bin_size_x / float(grid_size_x);

                // Interpolation in the [0,0] [0,1] [1,0] [1,1] square
                const int y_low  = y;
                const int x_low  = x;
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1. - ly;
                const float hx = 1. - lx;

                const float w1 = hy * hx;
                const float w2 = hy * lx;
                const float w3 = ly * hx;
                const float w4 = ly * lx;
                if(data_layout == DataLayout::NCHW)
                {
                    const auto data1 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_low, pz, roi_batch)));
                    const auto data2 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_low, pz, roi_batch)));
                    const auto data3 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_high, pz, roi_batch)));
                    const auto data4 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_high, pz, roi_batch)));
                    avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                }
                else
                {
                    const auto data1 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_low, roi_batch)));
                    const auto data2 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_low, roi_batch)));
                    const auto data3 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_high, roi_batch)));
                    const auto data4 = *reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_high, roi_batch)));
                    avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                }
            }
        }

        avg /= grid_size_x * grid_size_y;
        return input_data_type(avg);
    }
}

/** Average pooling over an aligned window */
template <typename input_data_type>
inline input_data_type roi_align_1x1_qasymm8(const ITensor          *input,
                                             unsigned int            roi_batch,
                                             float                   region_start_x,
                                             float                   bin_size_x,
                                             int                     grid_size_x,
                                             float                   region_end_x,
                                             float                   region_start_y,
                                             float                   bin_size_y,
                                             int                     grid_size_y,
                                             float                   region_end_y,
                                             int                     pz,
                                             const QuantizationInfo &out_qinfo)
{
    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
    {
        return input_data_type(out_qinfo.uniform().offset);
    }
    else
    {
        float                         avg              = 0;
        const UniformQuantizationInfo input_qinfo      = input->info()->quantization_info().uniform();
        const bool                    is_qasymm_signed = is_data_type_quantized_asymmetric_signed(input->info()->data_type());
        const DataLayout              data_layout      = input->info()->data_layout();

        // Iterate through the aligned pooling region
        for(int iy = 0; iy < grid_size_y; ++iy)
        {
            for(int ix = 0; ix < grid_size_x; ++ix)
            {
                // Align the window in the middle of every bin
                float y = region_start_y + (iy + 0.5) * bin_size_y / float(grid_size_y);
                float x = region_start_x + (ix + 0.5) * bin_size_x / float(grid_size_x);

                // Interpolation in the [0,0] [0,1] [1,0] [1,1] square
                const int y_low  = y;
                const int x_low  = x;
                const int y_high = y_low + 1;
                const int x_high = x_low + 1;

                const float ly = y - y_low;
                const float lx = x - x_low;
                const float hy = 1. - ly;
                const float hx = 1. - lx;

                const float w1 = hy * hx;
                const float w2 = hy * lx;
                const float w3 = ly * hx;
                const float w4 = ly * lx;

                if(data_layout == DataLayout::NCHW)
                {
                    if(is_qasymm_signed)
                    {
                        float data1 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_low, pz, roi_batch))), input_qinfo);
                        float data2 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_low, pz, roi_batch))), input_qinfo);
                        float data3 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_high, pz, roi_batch))), input_qinfo);
                        float data4 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_high, pz, roi_batch))), input_qinfo);
                        avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                    }
                    else
                    {
                        float data1 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_low, pz, roi_batch))), input_qinfo);
                        float data2 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_low, pz, roi_batch))), input_qinfo);
                        float data3 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_low, y_high, pz, roi_batch))), input_qinfo);
                        float data4 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(x_high, y_high, pz, roi_batch))), input_qinfo);
                        avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                    }
                }
                else
                {
                    if(is_qasymm_signed)
                    {
                        const auto data1 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_low, roi_batch))), input_qinfo);
                        const auto data2 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_low, roi_batch))), input_qinfo);
                        const auto data3 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_high, roi_batch))), input_qinfo);
                        const auto data4 = dequantize_qasymm8_signed(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_high, roi_batch))), input_qinfo);
                        avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                    }
                    else
                    {
                        const auto data1 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_low, roi_batch))), input_qinfo);
                        const auto data2 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_low, roi_batch))), input_qinfo);
                        const auto data3 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_low, y_high, roi_batch))), input_qinfo);
                        const auto data4 = dequantize_qasymm8(*reinterpret_cast<const input_data_type *>(input->ptr_to_element(Coordinates(pz, x_high, y_high, roi_batch))), input_qinfo);
                        avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
                    }
                }
            }
        }

        avg /= grid_size_x * grid_size_y;

        input_data_type res = 0;
        if(is_qasymm_signed)
        {
            res = quantize_qasymm8_signed(avg, out_qinfo);
        }
        else
        {
            res = quantize_qasymm8(avg, out_qinfo);
        }
        return res;
    }
}

inline float compute_region_coordinate(int p, float bin_size, float roi_anchor, float max_value)
{
    const float region_start = p * bin_size + roi_anchor;
    return utility::clamp(region_start, 0.0f, max_value);
}

void NEROIAlignLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    const DataLayout data_layout = _input->info()->data_layout();
    if(data_layout == DataLayout::NCHW || data_layout == DataLayout::NHWC)
    {
        switch(_input->info()->data_type())
        {
            case DataType::QASYMM8:
            {
                NEROIAlignLayerKernel::internal_run<uint8_t, uint16_t>(window, info);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            {
                NEROIAlignLayerKernel::internal_run<int8_t, uint16_t>(window, info);
                break;
            }
            case DataType::F32:
            {
                NEROIAlignLayerKernel::internal_run<float>(window, info);
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                NEROIAlignLayerKernel::internal_run<float16_t>(window, info);
                break;
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            default:
            {
                ARM_COMPUTE_ERROR("DataType not supported");
                break;
            }
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Invalid layout");
    }
}

template <typename input_data_type, typename roi_data_type>
void NEROIAlignLayerKernel::internal_run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const DataLayout data_layout    = _input->info()->data_layout();
    const size_t     values_per_roi = _rois->info()->dimension(0);

    const int roi_list_start = window.x().start();
    const int roi_list_end   = window.x().end();

    const unsigned int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int idx_depth  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const int input_width   = _input->info()->dimension(idx_width);
    const int input_height  = _input->info()->dimension(idx_height);
    const int input_chanels = _input->info()->dimension(idx_depth);
    const int pooled_w      = _pool_info.pooled_width();
    const int pooled_h      = _pool_info.pooled_height();

    const DataType data_type = _input->info()->data_type();
    const bool     is_qasymm = is_data_type_quantized_asymmetric(data_type);

    const auto             *rois_ptr   = reinterpret_cast<const roi_data_type *>(_rois->buffer());
    const QuantizationInfo &rois_qinfo = _rois->info()->quantization_info();
    for(int roi_indx = roi_list_start; roi_indx < roi_list_end; ++roi_indx)
    {
        const unsigned int roi_batch = rois_ptr[values_per_roi * roi_indx];

        roi_data_type qx1 = rois_ptr[values_per_roi * roi_indx + 1];
        roi_data_type qy1 = rois_ptr[values_per_roi * roi_indx + 2];
        roi_data_type qx2 = rois_ptr[values_per_roi * roi_indx + 3];
        roi_data_type qy2 = rois_ptr[values_per_roi * roi_indx + 4];
        float         x1(qx1);
        float         x2(qx2);
        float         y1(qy1);
        float         y2(qy2);
        if(is_qasymm)
        {
            x1 = dequantize_qasymm16(qx1, rois_qinfo);
            x2 = dequantize_qasymm16(qx2, rois_qinfo);
            y1 = dequantize_qasymm16(qy1, rois_qinfo);
            y2 = dequantize_qasymm16(qy2, rois_qinfo);
        }
        const float roi_anchor_x = x1 * _pool_info.spatial_scale();
        const float roi_anchor_y = y1 * _pool_info.spatial_scale();
        const float roi_dims_x   = std::max((x2 - x1) * _pool_info.spatial_scale(), 1.0f);
        const float roi_dims_y   = std::max((y2 - y1) * _pool_info.spatial_scale(), 1.0f);
        float       bin_size_x   = roi_dims_x / _pool_info.pooled_width();
        float       bin_size_y   = roi_dims_y / _pool_info.pooled_height();

        // Iterate through all feature maps
        for(int ch = 0; ch < input_chanels; ++ch)
        {
            // Iterate through all output pixels
            for(int py = 0; py < pooled_h; ++py)
            {
                for(int px = 0; px < pooled_w; ++px)
                {
                    const float     region_start_x = compute_region_coordinate(px, bin_size_x, roi_anchor_x, input_width);
                    const float     region_start_y = compute_region_coordinate(py, bin_size_y, roi_anchor_y, input_height);
                    const float     region_end_x   = compute_region_coordinate(px + 1, bin_size_x, roi_anchor_x, input_width);
                    const float     region_end_y   = compute_region_coordinate(py + 1, bin_size_y, roi_anchor_y, input_height);
                    const int       roi_bin_grid_x = (_pool_info.sampling_ratio() > 0) ? _pool_info.sampling_ratio() : int(ceil(bin_size_x));
                    const int       roi_bin_grid_y = (_pool_info.sampling_ratio() > 0) ? _pool_info.sampling_ratio() : int(ceil(bin_size_y));
                    input_data_type out_val(0);
                    if(is_qasymm)
                    {
                        out_val = roi_align_1x1_qasymm8<input_data_type>(
                                      _input, roi_batch, region_start_x, bin_size_x,
                                      roi_bin_grid_x, region_end_x, region_start_y, bin_size_y,
                                      roi_bin_grid_y, region_end_y, ch, _output->info()->quantization_info());
                    }
                    else
                    {
                        out_val = roi_align_1x1<input_data_type>(
                                      _input, roi_batch, region_start_x, bin_size_x,
                                      roi_bin_grid_x, region_end_x, region_start_y, bin_size_y,
                                      roi_bin_grid_y, region_end_y, ch);
                    }

                    if(data_layout == DataLayout::NCHW)
                    {
                        auto out_ptr = reinterpret_cast<input_data_type *>(_output->ptr_to_element(Coordinates(px, py, ch, roi_indx)));
                        *out_ptr     = out_val;
                    }
                    else
                    {
                        auto out_ptr = reinterpret_cast<input_data_type *>(_output->ptr_to_element(Coordinates(ch, px, py, roi_indx)));
                        *out_ptr     = out_val;
                    }
                }
            }
        }
    }
}
} // namespace arm_compute
