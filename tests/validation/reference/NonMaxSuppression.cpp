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
#include "Permute.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/Helpers.h"
#include <queue>

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
inline float get_elem_by_coordinate(const SimpleTensor<float> &tensor, Coordinates coord)
{
    return *static_cast<const float *>(tensor(coord));
}

// Return intersection-over-union overlap between boxes i and j
inline bool iou_greater_than_threshold(const SimpleTensor<float> &boxes, size_t i, size_t j, float iou_threshold)
{
    const float ymin_i = std::min<float>(get_elem_by_coordinate(boxes, Coordinates(0, i)), get_elem_by_coordinate(boxes, Coordinates(2, i)));
    const float xmin_i = std::min<float>(get_elem_by_coordinate(boxes, Coordinates(1, i)), get_elem_by_coordinate(boxes, Coordinates(3, i)));
    const float ymax_i = std::max<float>(get_elem_by_coordinate(boxes, Coordinates(0, i)), get_elem_by_coordinate(boxes, Coordinates(2, i)));
    const float xmax_i = std::max<float>(get_elem_by_coordinate(boxes, Coordinates(1, i)), get_elem_by_coordinate(boxes, Coordinates(3, i)));
    const float ymin_j = std::min<float>(get_elem_by_coordinate(boxes, Coordinates(0, j)), get_elem_by_coordinate(boxes, Coordinates(2, j)));
    const float xmin_j = std::min<float>(get_elem_by_coordinate(boxes, Coordinates(1, j)), get_elem_by_coordinate(boxes, Coordinates(3, j)));
    const float ymax_j = std::max<float>(get_elem_by_coordinate(boxes, Coordinates(0, j)), get_elem_by_coordinate(boxes, Coordinates(2, j)));
    const float xmax_j = std::max<float>(get_elem_by_coordinate(boxes, Coordinates(1, j)), get_elem_by_coordinate(boxes, Coordinates(3, j)));
    const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
    const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
    if(area_i <= 0 || area_j <= 0)
    {
        return false;
    }
    const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
    const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
    const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
    const float intersection_xmax = std::min<float>(xmax_i, xmax_j);
    const float intersection_area = std::max<float>(intersection_ymax - intersection_ymin, 0.0) * std::max<float>(intersection_xmax - intersection_xmin, 0.0);
    const float iou               = intersection_area / (area_i + area_j - intersection_area);
    return iou > iou_threshold;
}

} // namespace

SimpleTensor<int> non_max_suppression(const SimpleTensor<float> &bboxes, const SimpleTensor<float> &scores, SimpleTensor<int> &indices,
                                      unsigned int max_output_size, float score_threshold, float nms_threshold)
{
    const size_t       num_boxes   = bboxes.shape().y();
    const size_t       output_size = std::min(static_cast<size_t>(max_output_size), num_boxes);
    std::vector<float> scores_data(num_boxes);
    std::copy_n(scores.data(), num_boxes, scores_data.begin());

    using CandidateBox = std::pair<int /* index */, float /* score */>;
    auto cmp           = [](const CandidateBox bb0, const CandidateBox bb1)
    {
        return bb0.second < bb1.second;
    };

    std::priority_queue<CandidateBox, std::deque<CandidateBox>, decltype(cmp)> candidate_priority_queue(cmp);
    for(size_t i = 0; i < scores_data.size(); ++i)
    {
        if(scores_data[i] > score_threshold)
        {
            candidate_priority_queue.emplace(CandidateBox({ i, scores_data[i] }));
        }
    }

    std::vector<int>   selected;
    std::vector<float> selected_scores;
    CandidateBox       next_candidate;

    while(selected.size() < output_size && !candidate_priority_queue.empty())
    {
        next_candidate = candidate_priority_queue.top();
        candidate_priority_queue.pop();
        bool should_select = true;
        for(int j = selected.size() - 1; j >= 0; --j)
        {
            if(iou_greater_than_threshold(bboxes, next_candidate.first, selected[j], nms_threshold))
            {
                should_select = false;
                break;
            }
        }
        if(should_select)
        {
            selected.push_back(next_candidate.first);
            selected_scores.push_back(next_candidate.second);
        }
    }
    std::copy_n(selected.begin(), selected.size(), indices.data());
    return indices;
}

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
