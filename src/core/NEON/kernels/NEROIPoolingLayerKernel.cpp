/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEROIPoolingLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <cfloat>
#include <cmath>

namespace arm_compute
{
NEROIPoolingLayerKernel::NEROIPoolingLayerKernel()
    : _input(nullptr), _rois(nullptr), _output(nullptr), _pool_info(0, 0, 0.f)
{
}

void NEROIPoolingLayerKernel::configure(const ITensor *input, const ITensor *rois, ITensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, rois);

    //Validate arguments
    ARM_COMPUTE_ERROR_ON_NULLPTR(input->info(), rois->info(), output->info());
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rois, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON(rois->info()->dimension(0) != 5);
    ARM_COMPUTE_ERROR_ON(rois->info()->num_dimensions() > 2);
    ARM_COMPUTE_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_ERROR_ON((pool_info.pooled_width() == 0) || (pool_info.pooled_height() == 0));

    if(output->info()->total_size() != 0)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pool_info.pooled_width()) || (output->info()->dimension(1) != pool_info.pooled_height()));
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != output->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(rois->info()->dimension(1) != output->info()->dimension(3));
    }

    // Output auto initialization if not yet initialized
    TensorShape output_shape(pool_info.pooled_width(), pool_info.pooled_height(), input->info()->dimension(2), rois->info()->dimension(1));
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pool_info.pooled_width()) || (output->info()->dimension(1) != pool_info.pooled_height()));

    // Set instance variables
    _input     = input;
    _rois      = rois;
    _output    = output;
    _pool_info = pool_info;

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, rois->info()->dimension(1)));
    window.set(Window::DimY, Window::Dimension(0, 1));

    AccessWindowStatic input_access(input->info(),
                                    input->info()->valid_region().start(0),
                                    input->info()->valid_region().start(1),
                                    input->info()->valid_region().end(0),
                                    input->info()->valid_region().end(1));
    AccessWindowStatic output_access(output->info(), 0, 0, pool_info.pooled_width(), pool_info.pooled_height());

    ARM_COMPUTE_UNUSED(update_window_and_padding(window, input_access, output_access));
    output_access.set_valid_region(window, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    INEKernel::configure(window);
}

void NEROIPoolingLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const size_t values_per_roi = _rois->info()->dimension(0);

    const int   roi_list_start = window.x().start();
    const int   roi_list_end   = window.x().end();
    const int   width          = _input->info()->dimension(Window::DimX);
    const int   height         = _input->info()->dimension(Window::DimY);
    const int   fms            = _input->info()->dimension(Window::DimZ);
    const int   pooled_w       = _pool_info.pooled_width();
    const int   pooled_h       = _pool_info.pooled_height();
    const float spatial_scale  = _pool_info.spatial_scale();

    const auto *rois_ptr = reinterpret_cast<const uint16_t *>(_rois->buffer());

    for(int roi_indx = roi_list_start; roi_indx < roi_list_end; ++roi_indx)
    {
        const unsigned int roi_batch = rois_ptr[values_per_roi * roi_indx];
        const auto         x1        = rois_ptr[values_per_roi * roi_indx + 1];
        const auto         y1        = rois_ptr[values_per_roi * roi_indx + 2];
        const auto         x2        = rois_ptr[values_per_roi * roi_indx + 3];
        const auto         y2        = rois_ptr[values_per_roi * roi_indx + 4];

        // Scale ROI
        const int roi_anchor_x = support::cpp11::round(x1 * spatial_scale);
        const int roi_anchor_y = support::cpp11::round(y1 * spatial_scale);
        const int roi_width    = std::max(support::cpp11::round((x2 - x1) * spatial_scale), 1.f);
        const int roi_height   = std::max(support::cpp11::round((y2 - y1) * spatial_scale), 1.f);

        // Iterate through all feature maps
        for(int fm = 0; fm < fms; ++fm)
        {
            // Iterate through all output pixels
            for(int py = 0; py < pooled_h; ++py)
            {
                for(int px = 0; px < pooled_w; ++px)
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
                        *reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(px, py, fm, roi_indx))) = 0;
                    }
                    else
                    {
                        float curr_max = -FLT_MAX;
                        for(int j = region_start_y; j < region_end_y; ++j)
                        {
                            for(int i = region_start_x; i < region_end_x; ++i)
                            {
                                const auto val = *reinterpret_cast<const float *>(_input->ptr_to_element(Coordinates(i, j, fm, roi_batch)));
                                curr_max       = std::max(val, curr_max);
                            }
                        }
                        *reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(px, py, fm, roi_indx))) = curr_max;
                    }
                }
            }
        }
    }
}
} // namespace arm_compute