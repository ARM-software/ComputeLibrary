/*
 * Copyright (c) 2021 Arm Limited.
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

#include "ROIPoolingLayer.h"
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
template <>
SimpleTensor<float> roi_pool_layer(const SimpleTensor<float> &src, const SimpleTensor<uint16_t> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo)
{
    ARM_COMPUTE_UNUSED(output_qinfo);

    const size_t num_rois         = rois.shape()[1];
    const size_t values_per_roi   = rois.shape()[0];
    DataType     output_data_type = src.data_type();

    TensorShape         input_shape = src.shape();
    TensorShape         output_shape(pool_info.pooled_width(), pool_info.pooled_height(), src.shape()[2], num_rois);
    SimpleTensor<float> output(output_shape, output_data_type);

    const int   pooled_w      = pool_info.pooled_width();
    const int   pooled_h      = pool_info.pooled_height();
    const float spatial_scale = pool_info.spatial_scale();

    // get sizes of x and y dimensions in src tensor
    const int width  = src.shape()[0];
    const int height = src.shape()[1];

    // Move pointer across the fourth dimension
    const size_t input_stride_w  = input_shape[0] * input_shape[1] * input_shape[2];
    const size_t output_stride_w = output_shape[0] * output_shape[1] * output_shape[2];

    const auto *rois_ptr = reinterpret_cast<const uint16_t *>(rois.data());

    // Iterate through pixel width (X-Axis)
    for(size_t pw = 0; pw < num_rois; ++pw)
    {
        const unsigned int roi_batch = rois_ptr[values_per_roi * pw];
        const auto         x1        = rois_ptr[values_per_roi * pw + 1];
        const auto         y1        = rois_ptr[values_per_roi * pw + 2];
        const auto         x2        = rois_ptr[values_per_roi * pw + 3];
        const auto         y2        = rois_ptr[values_per_roi * pw + 4];

        //Iterate through pixel height (Y-Axis)
        for(size_t fm = 0; fm < input_shape[2]; ++fm)
        {
            // Iterate through regions of interest index
            for(size_t py = 0; py < pool_info.pooled_height(); ++py)
            {
                // Scale ROI
                const int roi_anchor_x = support::cpp11::round(x1 * spatial_scale);
                const int roi_anchor_y = support::cpp11::round(y1 * spatial_scale);
                const int roi_width    = std::max(support::cpp11::round((x2 - x1) * spatial_scale), 1.f);
                const int roi_height   = std::max(support::cpp11::round((y2 - y1) * spatial_scale), 1.f);

                // Iterate over feature map (Z axis)
                for(size_t px = 0; px < pool_info.pooled_width(); ++px)
                {
                    auto region_start_x = static_cast<int>(std::floor((static_cast<float>(px) / pooled_w) * roi_width));
                    auto region_end_x   = static_cast<int>(std::floor((static_cast<float>(px + 1) / pooled_w) * roi_width));
                    auto region_start_y = static_cast<int>(std::floor((static_cast<float>(py) / pooled_h) * roi_height));
                    auto region_end_y   = static_cast<int>(std::floor((static_cast<float>(py + 1) / pooled_h) * roi_height));

                    region_start_x = std::min(std::max(region_start_x + roi_anchor_x, 0), width);
                    region_end_x   = std::min(std::max(region_end_x + roi_anchor_x, 0), width);
                    region_start_y = std::min(std::max(region_start_y + roi_anchor_y, 0), height);
                    region_end_y   = std::min(std::max(region_end_y + roi_anchor_y, 0), height);

                    // Iterate through the pooling region
                    if((region_end_x <= region_start_x) || (region_end_y <= region_start_y))
                    {
                        /* Assign element in tensor 'output' at coordinates px, py, fm, roi_indx, to 0 */
                        auto out_ptr = output.data() + px + py * output_shape[0] + fm * output_shape[0] * output_shape[1] + pw * output_stride_w;
                        *out_ptr     = 0;
                    }
                    else
                    {
                        float curr_max = -std::numeric_limits<float>::max();
                        for(int j = region_start_y; j < region_end_y; ++j)
                        {
                            for(int i = region_start_x; i < region_end_x; ++i)
                            {
                                /* Retrieve element from input tensor at coordinates(i, j, fm, roi_batch) */
                                float in_element = *(src.data() + i + j * input_shape[0] + fm * input_shape[0] * input_shape[1] + roi_batch * input_stride_w);
                                curr_max         = std::max(in_element, curr_max);
                            }
                        }

                        /* Assign element in tensor 'output' at coordinates px, py, fm, roi_indx, to curr_max */
                        auto out_ptr = output.data() + px + py * output_shape[0] + fm * output_shape[0] * output_shape[1] + pw * output_stride_w;
                        *out_ptr     = curr_max;
                    }
                }
            }
        }
    }

    return output;
}

/*
    Template genericised method to allow calling of roi_pooling_layer with quantized 8 bit datatype
*/
template <>
SimpleTensor<uint8_t> roi_pool_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint16_t> &rois, const ROIPoolingLayerInfo &pool_info, const QuantizationInfo &output_qinfo)
{
    const SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>       dst_tmp = roi_pool_layer<float>(src_tmp, rois, pool_info, output_qinfo);
    SimpleTensor<uint8_t>     dst     = convert_to_asymmetric<uint8_t>(dst_tmp, output_qinfo);
    return dst;
}

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute