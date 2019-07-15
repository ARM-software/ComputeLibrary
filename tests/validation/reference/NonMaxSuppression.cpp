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
#include "NonMaxSuppression.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/Helpers.h"

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
using CandidateBox = std::pair<int /* index */, float /* score */>;
using Box          = std::tuple<float, float, float, float>;

inline float get_elem_by_coordinate(const SimpleTensor<float> &tensor, Coordinates coord)
{
    return *static_cast<const float *>(tensor(coord));
}

inline Box get_box(const SimpleTensor<float> &boxes, size_t id)
{
    return std::make_tuple(
               get_elem_by_coordinate(boxes, Coordinates(0, id)),
               get_elem_by_coordinate(boxes, Coordinates(1, id)),
               get_elem_by_coordinate(boxes, Coordinates(2, id)),
               get_elem_by_coordinate(boxes, Coordinates(3, id)));
}

// returns a pair (minX, minY)
inline std::pair<float, float> get_min_yx(Box b)
{
    return std::make_pair(
               std::min<float>(std::get<0>(b), std::get<2>(b)),
               std::min<float>(std::get<1>(b), std::get<3>(b)));
}
// returns a pair (maxX, maxY)
inline std::pair<float, float> get_max_yx(Box b)
{
    return std::make_pair(
               std::max<float>(std::get<0>(b), std::get<2>(b)),
               std::max<float>(std::get<1>(b), std::get<3>(b)));
}

inline float compute_size(const std::pair<float, float> &min, const std::pair<float, float> &max)
{
    return (max.first - min.first) * (max.second - min.second);
}

inline float compute_intersection(const std::pair<float, float> &b0_min, const std::pair<float, float> &b0_max,
                                  const std::pair<float, float> &b1_min, const std::pair<float, float> &b1_max, float b0_size, float b1_size)
{
    const float inter = std::max<float>(std::min<float>(b0_max.first, b1_max.first) - std::max<float>(b0_min.first, b1_min.first), 0.0f) * std::max<float>(std::min<float>(b0_max.second,
                        b1_max.second)
                        - std::max<float>(b0_min.second, b1_min.second),
                        0.0f);
    return inter / (b0_size + b1_size - inter);
}

inline bool reject_box(Box b0, Box b1, float threshold)
{
    const auto  b0_min  = get_min_yx(b0);
    const auto  b0_max  = get_max_yx(b0);
    const auto  b1_min  = get_min_yx(b1);
    const auto  b1_max  = get_max_yx(b1);
    const float b0_size = compute_size(b0_min, b0_max);
    const float b1_size = compute_size(b1_min, b1_max);
    if(b0_size <= 0.f || b1_size <= 0.f)
    {
        return false;
    }
    else
    {
        const float box_weight = compute_intersection(b0_min, b0_max, b1_min, b1_max, b0_size, b1_size);
        return box_weight > threshold;
    }
}

inline std::vector<CandidateBox> get_candidates(const SimpleTensor<float> &scores, float threshold)
{
    std::vector<CandidateBox> candidates_vector;
    for(int i = 0; i < scores.num_elements(); ++i)
    {
        if(scores[i] >= threshold)
        {
            const auto cb = CandidateBox({ i, scores[i] });
            candidates_vector.push_back(cb);
        }
    }
    std::stable_sort(candidates_vector.begin(), candidates_vector.end(), [](const CandidateBox bb0, const CandidateBox bb1)
    {
        return bb0.second > bb1.second;
    });
    return candidates_vector;
}

inline bool is_box_selected(const CandidateBox &cb, const SimpleTensor<float> &bboxes, std::vector<int> &selected_boxes, float threshold)
{
    for(int j = selected_boxes.size() - 1; j >= 0; --j)
    {
        const auto selected_box_jth   = get_box(bboxes, selected_boxes[j]);
        const auto candidate_box      = get_box(bboxes, cb.first);
        const bool candidate_rejected = reject_box(candidate_box, selected_box_jth, threshold);
        if(candidate_rejected)
        {
            return false;
        }
    }
    return true;
}
} // namespace

SimpleTensor<int> non_max_suppression(const SimpleTensor<float> &bboxes, const SimpleTensor<float> &scores, SimpleTensor<int> &indices,
                                      unsigned int max_output_size, float score_threshold, float nms_threshold)
{
    const size_t                    num_boxes         = bboxes.shape().y();
    const size_t                    output_size       = std::min(static_cast<size_t>(max_output_size), num_boxes);
    const std::vector<CandidateBox> candidates_vector = get_candidates(scores, score_threshold);
    std::vector<int>                selected;
    for(const auto c : candidates_vector)
    {
        if(selected.size() == output_size)
        {
            break;
        }
        if(is_box_selected(c, bboxes, selected, nms_threshold))
        {
            selected.push_back(c.first);
        }
    }
    std::copy_n(selected.begin(), selected.size(), indices.data());

    for(unsigned int i = selected.size(); i < max_output_size; ++i)
    {
        indices[i] = -1;
    }

    return indices;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
