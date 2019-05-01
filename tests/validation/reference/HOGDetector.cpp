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
#include "HOGDetector.h"

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
/** Computes the number of detection windows to iterate over in the feature vector. */
Size2D num_detection_windows(const TensorShape &shape, const Size2D &window_step, const HOGInfo &hog_info)
{
    const size_t num_block_strides_width  = hog_info.detection_window_size().width / hog_info.block_stride().width;
    const size_t num_block_strides_height = hog_info.detection_window_size().height / hog_info.block_stride().height;

    return Size2D{ floor_to_multiple(shape.x() - num_block_strides_width, window_step.width) + window_step.width,
                   floor_to_multiple(shape.y() - num_block_strides_height, window_step.height) + window_step.height };
}
} // namespace

template <typename T>
std::vector<DetectionWindow> hog_detector(const SimpleTensor<T> &src, const std::vector<T> &descriptor, unsigned int max_num_detection_windows,
                                          const HOGInfo &hog_info, const Size2D &detection_window_stride, float threshold, uint16_t idx_class)
{
    ARM_COMPUTE_ERROR_ON_MSG((detection_window_stride.width % hog_info.block_stride().width != 0),
                             "Detection window stride width must be multiple of block stride width");
    ARM_COMPUTE_ERROR_ON_MSG((detection_window_stride.height % hog_info.block_stride().height != 0),
                             "Detection window stride height must be multiple of block stride height");

    // Create vector for identifying each detection window
    std::vector<DetectionWindow> windows;

    // Calculate detection window step
    const Size2D window_step(detection_window_stride.width / hog_info.block_stride().width,
                             detection_window_stride.height / hog_info.block_stride().height);

    // Calculate number of detection windows
    const Size2D num_windows = num_detection_windows(src.shape(), window_step, hog_info);

    // Calculate detection window and row offsets in feature vector
    const size_t src_offset_x   = window_step.width * hog_info.num_bins() * hog_info.num_cells_per_block().area();
    const size_t src_offset_y   = window_step.height * hog_info.num_bins() * hog_info.num_cells_per_block().area() * src.shape().x();
    const size_t src_offset_row = src.num_channels() * src.shape().x();

    // Calculate detection window attributes
    const Size2D       num_block_positions_per_detection_window = hog_info.num_block_positions_per_image(hog_info.detection_window_size());
    const unsigned int num_bins_per_descriptor_x                = num_block_positions_per_detection_window.width * src.num_channels();
    const unsigned int num_blocks_per_descriptor_y              = num_block_positions_per_detection_window.height;

    ARM_COMPUTE_ERROR_ON((num_bins_per_descriptor_x * num_blocks_per_descriptor_y + 1) != hog_info.descriptor_size());

    size_t win_id = 0;

    // Traverse feature vector in detection window steps
    for(auto win_y = 0u, offset_y = 0u; win_y < num_windows.height; win_y += window_step.height, offset_y += src_offset_y)
    {
        for(auto win_x = 0u, offset_x = 0u; win_x < num_windows.width; win_x += window_step.width, offset_x += src_offset_x)
        {
            // Reset the score
            float score = 0.0f;

            // Traverse detection window
            for(auto y = 0u, offset_row = 0u; y < num_blocks_per_descriptor_y; ++y, offset_row += src_offset_row)
            {
                const int bin_offset = y * num_bins_per_descriptor_x;

                for(auto x = 0u; x < num_bins_per_descriptor_x; ++x)
                {
                    // Compute Linear SVM
                    const float a = src[x + offset_x + offset_y + offset_row];
                    const float b = descriptor[x + bin_offset];
                    score += a * b;
                }
            }

            // Add the bias. The bias is located at the position (descriptor_size() - 1)
            score += descriptor[num_bins_per_descriptor_x * num_blocks_per_descriptor_y];

            if(score > threshold)
            {
                DetectionWindow window;

                if(win_id++ < max_num_detection_windows)
                {
                    window.x         = win_x * hog_info.block_stride().width;
                    window.y         = win_y * hog_info.block_stride().height;
                    window.width     = hog_info.detection_window_size().width;
                    window.height    = hog_info.detection_window_size().height;
                    window.idx_class = idx_class;
                    window.score     = score;

                    windows.push_back(window);
                }
            }
        }
    }

    return windows;
}

template std::vector<DetectionWindow> hog_detector(const SimpleTensor<float> &src, const std::vector<float> &descriptor, unsigned int max_num_detection_windows,
                                                   const HOGInfo &hog_info, const Size2D &detection_window_stride, float threshold, uint16_t idx_class);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
