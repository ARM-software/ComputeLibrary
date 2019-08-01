/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CPP/kernels/CPPNonMaximumSuppressionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <list>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *output_indices, unsigned int max_output_size,
                          const float score_threshold, const float iou_threshold)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(bboxes, scores, output_indices);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bboxes, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_indices, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(bboxes->num_dimensions() > 2, "The bboxes tensor must be a 2-D float tensor of shape [4, num_boxes].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(scores->num_dimensions() > 1, "The scores tensor must be a 1-D float tensor of shape [num_boxes].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output_indices->num_dimensions() > 1, "The indices must be 1-D integer tensor of shape [M], where max_output_size <= M");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(bboxes, scores);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output_indices->dimension(0) == 0, "Indices tensor must be bigger than 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(max_output_size == 0, "Max size cannot be 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(iou_threshold < 0.f || iou_threshold > 1.f, "IOU threshold must be in [0,1]");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(score_threshold < 0.f || score_threshold > 1.f, "Score threshold must be in [0,1]");

    return Status{};
}
} // namespace

CPPNonMaximumSuppressionKernel::CPPNonMaximumSuppressionKernel()
    : _input_bboxes(nullptr), _input_scores(nullptr), _output_indices(nullptr), _max_output_size(0), _score_threshold(0.f), _iou_threshold(0.f), _num_boxes(0), _scores_above_thd_vector(),
      _indices_above_thd_vector(), _visited(), _sorted_indices()
{
}

void CPPNonMaximumSuppressionKernel::configure(
    const ITensor *input_bboxes, const ITensor *input_scores, ITensor *output_indices, unsigned int max_output_size,
    const float score_threshold, const float iou_threshold)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_bboxes, input_scores, output_indices);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input_bboxes->info(), input_scores->info(), output_indices->info(), max_output_size, score_threshold, iou_threshold));

    auto_init_if_empty(*output_indices->info(), TensorShape(max_output_size), 1, DataType::U8, QuantizationInfo());

    _input_bboxes    = input_bboxes;
    _input_scores    = input_scores;
    _output_indices  = output_indices;
    _score_threshold = score_threshold;
    _iou_threshold   = iou_threshold;
    _max_output_size = max_output_size;
    _num_boxes       = input_scores->info()->dimension(0);

    _scores_above_thd_vector.reserve(_num_boxes);
    _indices_above_thd_vector.reserve(_num_boxes);

    // Visited and sorted_indices are preallocated as num_boxes size, which is the maximum size possible
    // Will be used only N elements where N is the number of score above the threshold
    _visited.reserve(_num_boxes);
    _sorted_indices.reserve(_num_boxes);

    // Configure kernel window
    Window win = calculate_max_window(*output_indices->info(), Steps());

    // The CPPNonMaximumSuppressionKernel doesn't need padding so update_window_and_padding() can be skipped
    ICPPKernel::configure(win);
}

Status CPPNonMaximumSuppressionKernel::validate(
    const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *output_indices, unsigned int max_output_size,
    const float score_threshold, const float iou_threshold)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(bboxes, scores, output_indices, max_output_size, score_threshold, iou_threshold));
    return Status{};
}

void CPPNonMaximumSuppressionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    unsigned int num_above_thd = 0;
    for(unsigned int i = 0; i < _num_boxes; ++i)
    {
        const float score_i = *(reinterpret_cast<float *>(_input_scores->ptr_to_element(Coordinates(i))));
        if(score_i >= _score_threshold)
        {
            _indices_above_thd_vector.emplace_back(i);
            _scores_above_thd_vector.emplace_back(score_i);
            // Initialize respective index and visited
            _sorted_indices.emplace_back(num_above_thd);
            _visited.push_back(false);
            ++num_above_thd;
        }
    }

    // Sort selected indices based on scores
    std::sort(_sorted_indices.begin(),
              _sorted_indices.end(),
              [&](unsigned int first, unsigned int second)
    {
        return _scores_above_thd_vector[first] > _scores_above_thd_vector[second];
    });

    // Number of output is the minimum between max_detection and the scores above the threshold
    const unsigned int num_output = std::min(_max_output_size, num_above_thd);
    unsigned int       output_idx = 0;

    for(unsigned int i = 0; i < num_above_thd; ++i)
    {
        // Check if the output is full
        if(output_idx >= num_output)
        {
            break;
        }

        // Check if it was already visited, if not add it to the output and update the indices counter
        if(!_visited[_sorted_indices[i]])
        {
            *(reinterpret_cast<int *>(_output_indices->ptr_to_element(Coordinates(output_idx)))) = _indices_above_thd_vector[_sorted_indices[i]];
            ++output_idx;
        }
        else
        {
            continue;
        }

        // Once added one element at the output check if the next ones overlap and can be skipped
        for(unsigned int j = i + 1; j < num_above_thd; ++j)
        {
            if(!_visited[_sorted_indices[j]])
            {
                // Calculate IoU
                const unsigned int i_index = _indices_above_thd_vector[_sorted_indices[i]];
                const unsigned int j_index = _indices_above_thd_vector[_sorted_indices[j]];
                // Box-corner format: xmin, ymin, xmax, ymax
                const auto box_i_xmin = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(0, i_index))));
                const auto box_i_ymin = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(1, i_index))));
                const auto box_i_xmax = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(2, i_index))));
                const auto box_i_ymax = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(3, i_index))));

                const auto box_j_xmin = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(0, j_index))));
                const auto box_j_ymin = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(1, j_index))));
                const auto box_j_xmax = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(2, j_index))));
                const auto box_j_ymax = *(reinterpret_cast<float *>(_input_bboxes->ptr_to_element(Coordinates(3, j_index))));

                const float area_i = (box_i_xmax - box_i_xmin) * (box_i_ymax - box_i_ymin);
                const float area_j = (box_j_xmax - box_j_xmin) * (box_j_ymax - box_j_ymin);
                float       overlap;
                if(area_i <= 0 || area_j <= 0)
                {
                    overlap = 0.0f;
                }
                else
                {
                    const auto y_min_intersection = std::max<float>(box_i_ymin, box_j_ymin);
                    const auto x_min_intersection = std::max<float>(box_i_xmin, box_j_xmin);
                    const auto y_max_intersection = std::min<float>(box_i_ymax, box_j_ymax);
                    const auto x_max_intersection = std::min<float>(box_i_xmax, box_j_xmax);
                    const auto area_intersection  = std::max<float>(y_max_intersection - y_min_intersection, 0.0f) * std::max<float>(x_max_intersection - x_min_intersection, 0.0f);
                    overlap                       = area_intersection / (area_i + area_j - area_intersection);
                }

                if(overlap > _iou_threshold)
                {
                    _visited[_sorted_indices[j]] = true;
                }
            }
        }
    }
    // The output could be full but not the output indices tensor
    // Instead return values not valid we put -1
    for(; output_idx < _max_output_size; ++output_idx)
    {
        *(reinterpret_cast<int *>(_output_indices->ptr_to_element(Coordinates(output_idx)))) = -1;
    }
}
} // namespace arm_compute
