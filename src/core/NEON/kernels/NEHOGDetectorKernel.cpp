/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEHOGDetectorKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

NEHOGDetectorKernel::NEHOGDetectorKernel()
    : _input(nullptr), _detection_windows(), _hog_descriptor(nullptr), _bias(0.0f), _threshold(0.0f), _idx_class(0), _num_bins_per_descriptor_x(0), _num_blocks_per_descriptor_y(0), _block_stride_width(0),
      _block_stride_height(0), _detection_window_width(0), _detection_window_height(0), _max_num_detection_windows(0), _mutex()
{
}

void NEHOGDetectorKernel::configure(const ITensor *input, const IHOG *hog, IDetectionWindowArray *detection_windows, const Size2D &detection_window_stride, float threshold, uint16_t idx_class)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F32);
    ARM_COMPUTE_ERROR_ON(hog == nullptr);
    ARM_COMPUTE_ERROR_ON(detection_windows == nullptr);
    ARM_COMPUTE_ERROR_ON((detection_window_stride.width % hog->info()->block_stride().width) != 0);
    ARM_COMPUTE_ERROR_ON((detection_window_stride.height % hog->info()->block_stride().height) != 0);

    const Size2D &detection_window_size = hog->info()->detection_window_size();
    const Size2D &block_size            = hog->info()->block_size();
    const Size2D &block_stride          = hog->info()->block_stride();

    _input                       = input;
    _detection_windows           = detection_windows;
    _threshold                   = threshold;
    _idx_class                   = idx_class;
    _hog_descriptor              = hog->descriptor();
    _bias                        = _hog_descriptor[hog->info()->descriptor_size() - 1];
    _num_bins_per_descriptor_x   = ((detection_window_size.width - block_size.width) / block_stride.width + 1) * input->info()->num_channels();
    _num_blocks_per_descriptor_y = (detection_window_size.height - block_size.height) / block_stride.height + 1;
    _block_stride_width          = block_stride.width;
    _block_stride_height         = block_stride.height;
    _detection_window_width      = detection_window_size.width;
    _detection_window_height     = detection_window_size.height;
    _max_num_detection_windows   = detection_windows->max_num_values();

    ARM_COMPUTE_ERROR_ON((_num_bins_per_descriptor_x * _num_blocks_per_descriptor_y + 1) != hog->info()->descriptor_size());

    // Get the number of blocks along the x and y directions of the input tensor
    const ValidRegion &valid_region = input->info()->valid_region();
    const size_t       num_blocks_x = valid_region.shape[0];
    const size_t       num_blocks_y = valid_region.shape[1];

    // Get the number of blocks along the x and y directions of the detection window
    const size_t num_blocks_per_detection_window_x = detection_window_size.width / block_stride.width;
    const size_t num_blocks_per_detection_window_y = detection_window_size.height / block_stride.height;

    const size_t window_step_x = detection_window_stride.width / block_stride.width;
    const size_t window_step_y = detection_window_stride.height / block_stride.height;

    // Configure kernel window
    Window win;
    win.set(Window::DimX, Window::Dimension(0, floor_to_multiple(num_blocks_x - num_blocks_per_detection_window_x, window_step_x), window_step_x));
    win.set(Window::DimY, Window::Dimension(0, floor_to_multiple(num_blocks_y - num_blocks_per_detection_window_y, window_step_y), window_step_y));

    constexpr unsigned int num_elems_read_per_iteration = 1;
    const unsigned int     num_rows_read_per_iteration  = _num_blocks_per_descriptor_y;

    update_window_and_padding(win, AccessWindowRectangle(input->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration));

    INEKernel::configure(win);
}

void NEHOGDetectorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_hog_descriptor == nullptr);

    const size_t in_step_y = _input->info()->strides_in_bytes()[Window::DimY] / data_size_from_type(_input->info()->data_type());

    Iterator in(_input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto *in_row_ptr = reinterpret_cast<const float *>(in.ptr());

        // Init score_f32 with 0
        float32x4_t score_f32 = vdupq_n_f32(0.0f);

        // Init score with bias
        float score = _bias;

        // Compute Linear SVM
        for(size_t yb = 0; yb < _num_blocks_per_descriptor_y; ++yb, in_row_ptr += in_step_y)
        {
            int32_t xb = 0;

            const int32_t offset_y = yb * _num_bins_per_descriptor_x;

            for(; xb < static_cast<int32_t>(_num_bins_per_descriptor_x) - 16; xb += 16)
            {
                // Load descriptor values
                const float32x4x4_t a_f32 =
                {
                    {
                        vld1q_f32(&in_row_ptr[xb + 0]),
                        vld1q_f32(&in_row_ptr[xb + 4]),
                        vld1q_f32(&in_row_ptr[xb + 8]),
                        vld1q_f32(&in_row_ptr[xb + 12])
                    }
                };

                // Load detector values
                const float32x4x4_t b_f32 =
                {
                    {
                        vld1q_f32(&_hog_descriptor[xb + 0 + offset_y]),
                        vld1q_f32(&_hog_descriptor[xb + 4 + offset_y]),
                        vld1q_f32(&_hog_descriptor[xb + 8 + offset_y]),
                        vld1q_f32(&_hog_descriptor[xb + 12 + offset_y])
                    }
                };

                // Multiply accumulate
                score_f32 = vmlaq_f32(score_f32, a_f32.val[0], b_f32.val[0]);
                score_f32 = vmlaq_f32(score_f32, a_f32.val[1], b_f32.val[1]);
                score_f32 = vmlaq_f32(score_f32, a_f32.val[2], b_f32.val[2]);
                score_f32 = vmlaq_f32(score_f32, a_f32.val[3], b_f32.val[3]);
            }

            for(; xb < static_cast<int32_t>(_num_bins_per_descriptor_x); ++xb)
            {
                const float a = in_row_ptr[xb];
                const float b = _hog_descriptor[xb + offset_y];

                score += a * b;
            }
        }

        score += vgetq_lane_f32(score_f32, 0);
        score += vgetq_lane_f32(score_f32, 1);
        score += vgetq_lane_f32(score_f32, 2);
        score += vgetq_lane_f32(score_f32, 3);

        if(score > _threshold)
        {
            if(_detection_windows->num_values() < _max_num_detection_windows)
            {
                DetectionWindow win;
                win.x         = (id.x() * _block_stride_width);
                win.y         = (id.y() * _block_stride_height);
                win.width     = _detection_window_width;
                win.height    = _detection_window_height;
                win.idx_class = _idx_class;
                win.score     = score;

                std::unique_lock<arm_compute::Mutex> lock(_mutex);
                _detection_windows->push_back(win);
                lock.unlock();
            }
        }
    },
    in);
}
