/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "ROIAlignLayer.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/validation/Helpers.h"

#include <algorithm>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
/** Average pooling over an aligned window */
template <typename T>
inline T roi_align_1x1(const T *input, TensorShape input_shape,
                       float region_start_x,
                       float bin_size_x,
                       int   grid_size_x,
                       float region_end_x,
                       float region_start_y,
                       float bin_size_y,
                       int   grid_size_y,
                       float region_end_y,
                       int   pz)
{
    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
    {
        return T(0);
    }
    else
    {
        float avg = 0;
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

                const size_t idx1  = coord2index(input_shape, Coordinates(x_low, y_low, pz));
                T            data1 = input[idx1];

                const size_t idx2  = coord2index(input_shape, Coordinates(x_high, y_low, pz));
                T            data2 = input[idx2];

                const size_t idx3  = coord2index(input_shape, Coordinates(x_low, y_high, pz));
                T            data3 = input[idx3];

                const size_t idx4  = coord2index(input_shape, Coordinates(x_high, y_high, pz));
                T            data4 = input[idx4];

                avg += w1 * data1 + w2 * data2 + w3 * data3 + w4 * data4;
            }
        }

        avg /= grid_size_x * grid_size_y;

        return T(avg);
    }
}

/** Clamp the value between lower and upper */
template <typename T>
T clamp(T value, T lower, T upper)
{
    return std::max(lower, std::min(value, upper));
}

SimpleTensor<float> convert_rois_from_asymmetric(SimpleTensor<uint16_t> rois)
{
    const UniformQuantizationInfo &quantization_info = rois.quantization_info().uniform();
    SimpleTensor<float>            dst{ rois.shape(), DataType::F32, 1, QuantizationInfo(), rois.data_layout() };

    for(int i = 0; i < rois.num_elements(); i += 5)
    {
        dst[i]     = static_cast<float>(rois[i]); // batch idx
        dst[i + 1] = dequantize_qasymm16(rois[i + 1], quantization_info);
        dst[i + 2] = dequantize_qasymm16(rois[i + 2], quantization_info);
        dst[i + 3] = dequantize_qasymm16(rois[i + 3], quantization_info);
        dst[i + 4] = dequantize_qasymm16(rois[i + 4], quantization_info);
    }
    return dst;
}
} // namespace
template <typename T, typename TRois>
SimpleTensor<T> roi_align_layer(const SimpleTensor<T> &src, const SimpleTensor<TRois> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo)
{
    ARM_COMPUTE_UNUSED(output_qinfo);

    const size_t values_per_roi = rois.shape()[0];
    const size_t num_rois       = rois.shape()[1];
    DataType     dst_data_type  = src.data_type();

    const auto *rois_ptr = static_cast<const TRois *>(rois.data());

    TensorShape     input_shape = src.shape();
    TensorShape     output_shape(pool_info.pooled_width(), pool_info.pooled_height(), src.shape()[2], num_rois);
    SimpleTensor<T> dst(output_shape, dst_data_type);

    // Iterate over every pixel of the input image
    for(size_t px = 0; px < pool_info.pooled_width(); ++px)
    {
        for(size_t py = 0; py < pool_info.pooled_height(); ++py)
        {
            for(size_t pw = 0; pw < num_rois; ++pw)
            {
                const unsigned int roi_batch = rois_ptr[values_per_roi * pw];
                const auto         x1        = float(rois_ptr[values_per_roi * pw + 1]);
                const auto         y1        = float(rois_ptr[values_per_roi * pw + 2]);
                const auto         x2        = float(rois_ptr[values_per_roi * pw + 3]);
                const auto         y2        = float(rois_ptr[values_per_roi * pw + 4]);

                const float roi_anchor_x = x1 * pool_info.spatial_scale();
                const float roi_anchor_y = y1 * pool_info.spatial_scale();
                const float roi_dims_x   = std::max((x2 - x1) * pool_info.spatial_scale(), 1.0f);
                const float roi_dims_y   = std::max((y2 - y1) * pool_info.spatial_scale(), 1.0f);

                float bin_size_x     = roi_dims_x / pool_info.pooled_width();
                float bin_size_y     = roi_dims_y / pool_info.pooled_height();
                float region_start_x = px * bin_size_x + roi_anchor_x;
                float region_start_y = py * bin_size_y + roi_anchor_y;
                float region_end_x   = (px + 1) * bin_size_x + roi_anchor_x;
                float region_end_y   = (py + 1) * bin_size_y + roi_anchor_y;

                region_start_x = clamp(region_start_x, 0.0f, float(input_shape[0]));
                region_start_y = clamp(region_start_y, 0.0f, float(input_shape[1]));
                region_end_x   = clamp(region_end_x, 0.0f, float(input_shape[0]));
                region_end_y   = clamp(region_end_y, 0.0f, float(input_shape[1]));

                const int roi_bin_grid_x = (pool_info.sampling_ratio() > 0) ? pool_info.sampling_ratio() : int(ceil(bin_size_x));
                const int roi_bin_grid_y = (pool_info.sampling_ratio() > 0) ? pool_info.sampling_ratio() : int(ceil(bin_size_y));

                // Move input and output pointer across the fourth dimension
                const size_t input_stride_w  = input_shape[0] * input_shape[1] * input_shape[2];
                const size_t output_stride_w = output_shape[0] * output_shape[1] * output_shape[2];
                const T     *input_ptr       = src.data() + roi_batch * input_stride_w;
                T           *output_ptr      = dst.data() + px + py * output_shape[0] + pw * output_stride_w;

                for(int pz = 0; pz < int(input_shape[2]); ++pz)
                {
                    // For every pixel pool over an aligned region
                    *(output_ptr + pz * output_shape[0] * output_shape[1]) = roi_align_1x1(input_ptr, input_shape,
                                                                                           region_start_x,
                                                                                           bin_size_x,
                                                                                           roi_bin_grid_x,
                                                                                           region_end_x,
                                                                                           region_start_y,
                                                                                           bin_size_y,
                                                                                           roi_bin_grid_y,
                                                                                           region_end_y, pz);
                }
            }
        }
    }
    return dst;
}

template SimpleTensor<float> roi_align_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo);
template SimpleTensor<half> roi_align_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo);

template <>
SimpleTensor<uint8_t> roi_align_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint16_t> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo)
{
    SimpleTensor<float>   src_tmp  = convert_from_asymmetric(src);
    SimpleTensor<float>   rois_tmp = convert_rois_from_asymmetric(rois);
    SimpleTensor<float>   dst_tmp  = roi_align_layer<float, float>(src_tmp, rois_tmp, pool_info, output_qinfo);
    SimpleTensor<uint8_t> dst      = convert_to_asymmetric<uint8_t>(dst_tmp, output_qinfo);
    return dst;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
