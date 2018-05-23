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
#include "HOGMultiDetection.h"

#include "Derivative.h"
#include "HOGDescriptor.h"
#include "HOGDetector.h"
#include "Magnitude.h"
#include "Phase.h"

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
void validate_models(const std::vector<HOGInfo> &models)
{
    ARM_COMPUTE_ERROR_ON(0 == models.size());

    for(size_t i = 1; i < models.size(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_MSG(models[0].phase_type() != models[i].phase_type(),
                                 "All HOG parameters must have the same phase type");

        ARM_COMPUTE_ERROR_ON_MSG(models[0].normalization_type() != models[i].normalization_type(),
                                 "All HOG parameters must have the same normalization_type");

        ARM_COMPUTE_ERROR_ON_MSG((models[0].l2_hyst_threshold() != models[i].l2_hyst_threshold()) && (models[0].normalization_type() == arm_compute::HOGNormType::L2HYS_NORM),
                                 "All HOG parameters must have the same l2 hysteresis threshold if you use L2 hysteresis normalization type");
    }
}
} // namespace

void detection_windows_non_maxima_suppression(std::vector<DetectionWindow> &multi_windows, float min_distance)
{
    const size_t num_candidates = multi_windows.size();
    size_t       num_detections = 0;

    // Sort by idx_class first and by score second
    std::sort(multi_windows.begin(), multi_windows.end(), [](const DetectionWindow & lhs, const DetectionWindow & rhs)
    {
        if(lhs.idx_class < rhs.idx_class)
        {
            return true;
        }
        if(rhs.idx_class < lhs.idx_class)
        {
            return false;
        }

        // idx_classes are equal so compare by score
        if(lhs.score > rhs.score)
        {
            return true;
        }
        if(rhs.score > lhs.score)
        {
            return false;
        }

        return false;
    });

    const float min_distance_pow2 = min_distance * min_distance;

    // Euclidean distance
    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(0.0f != multi_windows.at(i).score)
        {
            DetectionWindow cur;
            cur.x         = multi_windows.at(i).x;
            cur.y         = multi_windows.at(i).y;
            cur.width     = multi_windows.at(i).width;
            cur.height    = multi_windows.at(i).height;
            cur.idx_class = multi_windows.at(i).idx_class;
            cur.score     = multi_windows.at(i).score;

            // Store window
            multi_windows.at(num_detections) = cur;
            ++num_detections;

            const float xc = cur.x + cur.width * 0.5f;
            const float yc = cur.y + cur.height * 0.5f;

            for(size_t k = i + 1; k < (num_candidates) && (cur.idx_class == multi_windows.at(k).idx_class); ++k)
            {
                const float xn = multi_windows.at(k).x + multi_windows.at(k).width * 0.5f;
                const float yn = multi_windows.at(k).y + multi_windows.at(k).height * 0.5f;

                const float dx = std::fabs(xn - xc);
                const float dy = std::fabs(yn - yc);

                if(dx < min_distance && dy < min_distance)
                {
                    const float d = dx * dx + dy * dy;

                    if(d < min_distance_pow2)
                    {
                        // Invalidate detection window
                        multi_windows.at(k).score = 0.0f;
                    }
                }
            }
        }
    }

    multi_windows.resize(num_detections);
}

template <typename T>
std::vector<DetectionWindow> hog_multi_detection(const SimpleTensor<T> &src, BorderMode border_mode, T constant_border_value,
                                                 const std::vector<HOGInfo> &models, std::vector<std::vector<float>> descriptors,
                                                 unsigned int max_num_detection_windows, float threshold, bool non_maxima_suppression, float min_distance)
{
    ARM_COMPUTE_ERROR_ON(descriptors.size() != models.size());
    validate_models(models);

    const size_t width      = src.shape().x();
    const size_t height     = src.shape().y();
    const size_t num_models = models.size();

    // Initialize previous values
    size_t prev_num_bins     = models[0].num_bins();
    Size2D prev_cell_size    = models[0].cell_size();
    Size2D prev_block_size   = models[0].block_size();
    Size2D prev_block_stride = models[0].block_stride();

    std::vector<size_t> input_orient_bin;
    std::vector<size_t> input_hog_detect;
    std::vector<std::pair<size_t, size_t>> input_block_norm;

    input_orient_bin.push_back(0);
    input_hog_detect.push_back(0);
    input_block_norm.emplace_back(0, 0);

    // Iterate through the number of models and check if orientation binning
    // and block normalization steps can be skipped
    for(size_t i = 1; i < num_models; ++i)
    {
        size_t cur_num_bins     = models[i].num_bins();
        Size2D cur_cell_size    = models[i].cell_size();
        Size2D cur_block_size   = models[i].block_size();
        Size2D cur_block_stride = models[i].block_stride();

        // Check if binning and normalization steps are required
        if((cur_num_bins != prev_num_bins) || (cur_cell_size.width != prev_cell_size.width) || (cur_cell_size.height != prev_cell_size.height))
        {
            prev_num_bins     = cur_num_bins;
            prev_cell_size    = cur_cell_size;
            prev_block_size   = cur_block_size;
            prev_block_stride = cur_block_stride;

            // Compute orientation binning and block normalization. Update input to process
            input_orient_bin.push_back(i);
            input_block_norm.emplace_back(i, input_orient_bin.size() - 1);
        }
        else if((cur_block_size.width != prev_block_size.width) || (cur_block_size.height != prev_block_size.height) || (cur_block_stride.width != prev_block_stride.width)
                || (cur_block_stride.height != prev_block_stride.height))
        {
            prev_block_size   = cur_block_size;
            prev_block_stride = cur_block_stride;

            // Compute block normalization. Update input to process
            input_block_norm.emplace_back(i, input_orient_bin.size() - 1);
        }

        // Update input to process for hog detector
        input_hog_detect.push_back(input_block_norm.size() - 1);
    }

    size_t num_orient_bin = input_orient_bin.size();
    size_t num_block_norm = input_block_norm.size();
    size_t num_hog_detect = input_hog_detect.size();

    std::vector<SimpleTensor<float>> hog_spaces(num_orient_bin);
    std::vector<SimpleTensor<float>> hog_norm_spaces(num_block_norm);

    // Calculate derivative
    SimpleTensor<int16_t> grad_x;
    SimpleTensor<int16_t> grad_y;
    std::tie(grad_x, grad_y) = derivative<int16_t>(src, border_mode, constant_border_value, GradientDimension::GRAD_XY);

    // Calculate magnitude and phase
    SimpleTensor<int16_t> _mag   = magnitude(grad_x, grad_y, MagnitudeType::L2NORM);
    SimpleTensor<uint8_t> _phase = phase(grad_x, grad_y, models[0].phase_type());

    // Calculate Tensors for the HOG space and orientation binning
    for(size_t i = 0; i < num_orient_bin; ++i)
    {
        const size_t idx_multi_hog = input_orient_bin[i];

        const size_t num_bins    = models[idx_multi_hog].num_bins();
        const size_t num_cells_x = width / models[idx_multi_hog].cell_size().width;
        const size_t num_cells_y = height / models[idx_multi_hog].cell_size().height;

        // TensorShape of hog space
        TensorShape hog_space_shape(num_cells_x, num_cells_y);

        // Initialise HOG space
        TensorInfo info_hog_space(hog_space_shape, num_bins, DataType::F32);
        hog_spaces.at(i) = SimpleTensor<float>(info_hog_space.tensor_shape(), DataType::F32, info_hog_space.num_channels());

        // For each cell create histogram based on magnitude and phase
        hog_orientation_binning(_mag, _phase, hog_spaces[i], models[idx_multi_hog]);
    }

    // Calculate Tensors for the normalized HOG space and block normalization
    for(size_t i = 0; i < num_block_norm; ++i)
    {
        const size_t idx_multi_hog  = input_block_norm[i].first;
        const size_t idx_orient_bin = input_block_norm[i].second;

        // Create tensor info for HOG descriptor
        TensorInfo tensor_info(models[idx_multi_hog], src.shape().x(), src.shape().y());
        hog_norm_spaces.at(i) = SimpleTensor<float>(tensor_info.tensor_shape(), DataType::F32, tensor_info.num_channels());

        // Normalize histograms based on block size
        hog_block_normalization(hog_norm_spaces[i], hog_spaces[idx_orient_bin], models[idx_multi_hog]);
    }

    std::vector<DetectionWindow> multi_windows;

    // Calculate Detection Windows for HOG detector
    for(size_t i = 0; i < num_hog_detect; ++i)
    {
        const size_t idx_block_norm = input_hog_detect[i];

        // NOTE: Detection window stride fixed to block stride
        const Size2D detection_window_stride = models[i].block_stride();

        std::vector<DetectionWindow> windows = hog_detector(hog_norm_spaces[idx_block_norm], descriptors[i],
                                                            max_num_detection_windows, models[i], detection_window_stride, threshold, i);

        multi_windows.insert(multi_windows.end(), windows.begin(), windows.end());
    }

    // Suppress Non-maxima detection windows
    if(non_maxima_suppression)
    {
        detection_windows_non_maxima_suppression(multi_windows, min_distance);
    }

    return multi_windows;
}

template std::vector<DetectionWindow> hog_multi_detection(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value,
                                                          const std::vector<HOGInfo> &models, std::vector<std::vector<float>> descriptors,
                                                          unsigned int max_num_detection_windows, float threshold, bool non_maxima_suppression, float min_distance);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
