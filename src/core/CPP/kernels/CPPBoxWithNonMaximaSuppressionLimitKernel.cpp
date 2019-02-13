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
#include "arm_compute/core/CPP/kernels/CPPBoxWithNonMaximaSuppressionLimitKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"

#include <algorithm>
#include <cmath>

namespace arm_compute
{
namespace
{
template <typename T>
std::vector<int> SoftNMS(const ITensor *proposals, std::vector<std::vector<T>> &scores_in, std::vector<int> inds, const BoxNMSLimitInfo &info, int class_id)
{
    std::vector<int> keep;
    const int        proposals_width = proposals->info()->dimension(1);

    std::vector<T> x1(proposals_width);
    std::vector<T> y1(proposals_width);
    std::vector<T> x2(proposals_width);
    std::vector<T> y2(proposals_width);
    std::vector<T> areas(proposals_width);

    for(int i = 0; i < proposals_width; ++i)
    {
        x1[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4, i)));
        y1[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 1, i)));
        x2[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 2, i)));
        y2[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 3, i)));
        areas[i] = (x2[i] - x1[i] + 1.0) * (y2[i] - y1[i] + 1.0);
    }

    // Note: Soft NMS scores have already been initialized with input scores

    while(!inds.empty())
    {
        // Find proposal with max score among remaining proposals
        int max_pos = 0;
        for(unsigned int i = 1; i < inds.size(); ++i)
        {
            if(scores_in[class_id][inds.at(i)] > scores_in[class_id][inds.at(max_pos)])
            {
                max_pos = i;
            }
        }
        int element = inds.at(max_pos);
        keep.push_back(element);
        std::swap(inds.at(0), inds.at(max_pos));

        // Remove first element and compute IoU of the remaining boxes with identified max box
        inds.erase(inds.begin());

        std::vector<int> sorted_indices_temp;
        for(auto idx : inds)
        {
            const auto xx1 = std::max(x1[idx], x1[element]);
            const auto yy1 = std::max(y1[idx], y1[element]);
            const auto xx2 = std::min(x2[idx], x2[element]);
            const auto yy2 = std::min(y2[idx], y2[element]);

            const auto w     = std::max((xx2 - xx1 + 1.f), 0.f);
            const auto h     = std::max((yy2 - yy1 + 1.f), 0.f);
            const auto inter = w * h;
            const auto ovr   = inter / (areas[element] + areas[idx] - inter);

            // Update scores based on computed IoU, overlap threshold and NMS method
            T weight;
            switch(info.soft_nms_method())
            {
                case NMSType::LINEAR:
                    weight = (ovr > info.nms()) ? (1.f - ovr) : 1.f;
                    break;
                case NMSType::GAUSSIAN: // Gaussian
                    weight = std::exp(-1.f * ovr * ovr / info.soft_nms_sigma());
                    break;
                case NMSType::ORIGINAL: // Original NMS
                    weight = (ovr > info.nms()) ? 0.f : 1.f;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
            }

            // Discard boxes with new scores below min threshold and update pending indices
            scores_in[class_id][idx] *= weight;
            if(scores_in[class_id][idx] >= info.soft_nms_min_score_thres())
            {
                sorted_indices_temp.push_back(idx);
            }
        }
        inds = sorted_indices_temp;
    }

    return keep;
}

template <typename T>
std::vector<int> NonMaximaSuppression(const ITensor *proposals, std::vector<int> sorted_indices, const BoxNMSLimitInfo &info, int class_id)
{
    std::vector<int> keep;

    const int proposals_width = proposals->info()->dimension(1);

    std::vector<T> x1(proposals_width);
    std::vector<T> y1(proposals_width);
    std::vector<T> x2(proposals_width);
    std::vector<T> y2(proposals_width);
    std::vector<T> areas(proposals_width);

    for(int i = 0; i < proposals_width; ++i)
    {
        x1[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4, i)));
        y1[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 1, i)));
        x2[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 2, i)));
        y2[i]    = *reinterpret_cast<T *>(proposals->ptr_to_element(Coordinates(class_id * 4 + 3, i)));
        areas[i] = (x2[i] - x1[i] + 1.0) * (y2[i] - y1[i] + 1.0);
    }

    while(!sorted_indices.empty())
    {
        int i = sorted_indices.at(0);
        keep.push_back(i);

        std::vector<int> sorted_indices_temp = sorted_indices;
        std::vector<int> new_indices;
        sorted_indices_temp.erase(sorted_indices_temp.begin());

        for(unsigned int j = 0; j < sorted_indices_temp.size(); ++j)
        {
            const float xx1 = std::max(x1[sorted_indices_temp.at(j)], x1[i]);
            const float yy1 = std::max(y1[sorted_indices_temp.at(j)], y1[i]);
            const float xx2 = std::min(x2[sorted_indices_temp.at(j)], x2[i]);
            const float yy2 = std::min(y2[sorted_indices_temp.at(j)], y2[i]);

            const float w     = std::max((xx2 - xx1 + 1.f), 0.f);
            const float h     = std::max((yy2 - yy1 + 1.f), 0.f);
            const float inter = w * h;
            const float ovr   = inter / (areas[i] + areas[sorted_indices_temp.at(j)] - inter);
            const float ctr_x = xx1 + (w / 2);
            const float ctr_y = yy1 + (h / 2);

            // If suppress_size is specified, filter the boxes based on their size and position
            const bool keep_size = !info.suppress_size() || (w >= info.min_size() && h >= info.min_size() && ctr_x < info.im_width() && ctr_y < info.im_height());
            if(ovr <= info.nms() && keep_size)
            {
                new_indices.push_back(j);
            }
        }

        const unsigned int new_indices_size = new_indices.size();
        std::vector<int>   new_sorted_indices(new_indices_size);
        for(unsigned int i = 0; i < new_indices_size; ++i)
        {
            new_sorted_indices[i] = sorted_indices[new_indices[i] + 1];
        }
        sorted_indices = new_sorted_indices;
    }

    return keep;
}
} // namespace

CPPBoxWithNonMaximaSuppressionLimitKernel::CPPBoxWithNonMaximaSuppressionLimitKernel()
    : _scores_in(nullptr), _boxes_in(nullptr), _batch_splits_in(nullptr), _scores_out(nullptr), _boxes_out(nullptr), _classes(nullptr), _batch_splits_out(nullptr), _keeps(nullptr), _keeps_size(nullptr),
      _info()
{
}

bool CPPBoxWithNonMaximaSuppressionLimitKernel::is_parallelisable() const
{
    return false;
}

template <typename T>
void CPPBoxWithNonMaximaSuppressionLimitKernel::run_nmslimit()
{
    const int                     batch_size   = _batch_splits_in == nullptr ? 1 : _batch_splits_in->info()->dimension(0);
    const int                     num_classes  = _scores_in->info()->dimension(0);
    const int                     scores_count = _scores_in->info()->dimension(1);
    std::vector<int>              total_keep_per_batch(batch_size);
    std::vector<std::vector<int>> keeps(num_classes);
    int                           total_keep_count = 0;

    std::vector<std::vector<T>> in_scores(num_classes, std::vector<T>(scores_count));
    for(int i = 0; i < scores_count; ++i)
    {
        for(int j = 0; j < num_classes; ++j)
        {
            in_scores[j][i] = *reinterpret_cast<const T *>(_scores_in->ptr_to_element(Coordinates(j, i)));
        }
    }

    int offset        = 0;
    int cur_start_idx = 0;
    for(int b = 0; b < batch_size; ++b)
    {
        const int num_boxes = _batch_splits_in == nullptr ? 1 : static_cast<int>(*reinterpret_cast<T *>(_batch_splits_in->ptr_to_element(Coordinates(b))));
        // Skip first class if there is more than 1 except if the number of classes is 1.
        const int j_start = (num_classes == 1 ? 0 : 1);
        for(int j = j_start; j < num_classes; ++j)
        {
            std::vector<T>   cur_scores(scores_count);
            std::vector<int> inds;
            for(int i = 0; i < scores_count; ++i)
            {
                const T score = in_scores[j][i];
                cur_scores[i] = score;

                if(score > _info.score_thresh())
                {
                    inds.push_back(i);
                }
            }
            if(_info.soft_nms_enabled())
            {
                keeps[j] = SoftNMS(_boxes_in, in_scores, inds, _info, j);
            }
            else
            {
                std::sort(inds.data(), inds.data() + inds.size(),
                          [&cur_scores](int lhs, int rhs)
                {
                    return cur_scores[lhs] > cur_scores[rhs];
                });

                keeps[j] = NonMaximaSuppression<T>(_boxes_in, inds, _info, j);
            }
            total_keep_count += keeps[j].size();
        }

        if(_info.detections_per_im() > 0 && total_keep_count > _info.detections_per_im())
        {
            // merge all scores (represented by indices) together and sort
            auto get_all_scores_sorted = [&in_scores, &keeps, total_keep_count]()
            {
                std::vector<T> ret(total_keep_count);

                int ret_idx = 0;
                for(unsigned int i = 1; i < keeps.size(); ++i)
                {
                    auto &cur_keep = keeps[i];
                    for(auto &ckv : cur_keep)
                    {
                        ret[ret_idx++] = in_scores[i][ckv];
                    }
                }

                std::sort(ret.data(), ret.data() + ret.size());

                return ret;
            };

            auto    all_scores_sorted = get_all_scores_sorted();
            const T image_thresh      = all_scores_sorted[all_scores_sorted.size() - _info.detections_per_im()];
            for(int j = 1; j < num_classes; ++j)
            {
                auto            &cur_keep = keeps[j];
                std::vector<int> new_keeps_j;
                for(auto &k : cur_keep)
                {
                    if(in_scores[j][k] >= image_thresh)
                    {
                        new_keeps_j.push_back(k);
                    }
                }
                keeps[j] = new_keeps_j;
            }
            total_keep_count = _info.detections_per_im();
        }

        total_keep_per_batch[b] = total_keep_count;

        // Write results
        int cur_out_idx = 0;
        for(int j = j_start; j < num_classes; ++j)
        {
            auto     &cur_keep        = keeps[j];
            auto      cur_out_scores  = reinterpret_cast<T *>(_scores_out->ptr_to_element(Coordinates(cur_start_idx + cur_out_idx)));
            auto      cur_out_classes = reinterpret_cast<T *>(_classes->ptr_to_element(Coordinates(cur_start_idx + cur_out_idx)));
            const int box_column      = (cur_start_idx + cur_out_idx) * 4;

            for(unsigned int k = 0; k < cur_keep.size(); ++k)
            {
                cur_out_scores[k]     = in_scores[j][cur_keep[k]];
                cur_out_classes[k]    = static_cast<T>(j);
                auto cur_out_box_row0 = reinterpret_cast<T *>(_boxes_out->ptr_to_element(Coordinates(box_column + 0, k)));
                auto cur_out_box_row1 = reinterpret_cast<T *>(_boxes_out->ptr_to_element(Coordinates(box_column + 1, k)));
                auto cur_out_box_row2 = reinterpret_cast<T *>(_boxes_out->ptr_to_element(Coordinates(box_column + 2, k)));
                auto cur_out_box_row3 = reinterpret_cast<T *>(_boxes_out->ptr_to_element(Coordinates(box_column + 3, k)));
                *cur_out_box_row0     = *reinterpret_cast<const T *>(_boxes_in->ptr_to_element(Coordinates(j * 4 + 0, cur_keep[k])));
                *cur_out_box_row1     = *reinterpret_cast<const T *>(_boxes_in->ptr_to_element(Coordinates(j * 4 + 1, cur_keep[k])));
                *cur_out_box_row2     = *reinterpret_cast<const T *>(_boxes_in->ptr_to_element(Coordinates(j * 4 + 2, cur_keep[k])));
                *cur_out_box_row3     = *reinterpret_cast<const T *>(_boxes_in->ptr_to_element(Coordinates(j * 4 + 3, cur_keep[k])));
            }

            cur_out_idx += cur_keep.size();
        }

        if(_keeps != nullptr)
        {
            cur_out_idx = 0;
            for(int j = 0; j < num_classes; ++j)
            {
                for(unsigned int i = 0; i < keeps[j].size(); ++i)
                {
                    *reinterpret_cast<T *>(_keeps->ptr_to_element(Coordinates(cur_start_idx + cur_out_idx + i))) = static_cast<T>(keeps[j].at(i));
                }
                *reinterpret_cast<uint32_t *>(_keeps_size->ptr_to_element(Coordinates(j + b * num_classes))) = keeps[j].size();
                cur_out_idx += keeps[j].size();
            }
        }

        offset += num_boxes;
        cur_start_idx += total_keep_count;
    }

    if(_batch_splits_out != nullptr)
    {
        for(int b = 0; b < batch_size; ++b)
        {
            *reinterpret_cast<float *>(_batch_splits_out->ptr_to_element(Coordinates(b))) = total_keep_per_batch[b];
        }
    }
}

void CPPBoxWithNonMaximaSuppressionLimitKernel::configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                                                          ITensor *batch_splits_out, ITensor *keeps, ITensor *keeps_size, const BoxNMSLimitInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scores_in, boxes_in, scores_out, boxes_out, classes);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scores_in, 1, DataType::F16, DataType::F32);
    const unsigned int num_classes = scores_in->info()->dimension(0);

    ARM_COMPUTE_UNUSED(num_classes);
    ARM_COMPUTE_ERROR_ON_MSG((4 * num_classes) != boxes_in->info()->dimension(0), "First dimension of input boxes must be of size 4*num_classes");
    ARM_COMPUTE_ERROR_ON_MSG(scores_in->info()->dimension(1) != boxes_in->info()->dimension(1), "Input scores and input boxes must have the same number of rows");

    ARM_COMPUTE_ERROR_ON(scores_out->info()->dimension(0) != boxes_out->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(boxes_out->info()->dimension(0) != 4);
    if(keeps != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MSG(keeps_size == nullptr, "keeps_size cannot be nullptr if keeps has to be provided as output");
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(scores_in, keeps);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(keeps_size, 1, DataType::U32);
        ARM_COMPUTE_ERROR_ON(scores_out->info()->dimension(0) != keeps->info()->dimension(0));
        ARM_COMPUTE_ERROR_ON(num_classes != keeps_size->info()->dimension(0));
    }
    if(batch_splits_in != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(scores_in, batch_splits_in);
    }
    if(batch_splits_out != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(scores_in, batch_splits_out);
    }

    _scores_in        = scores_in;
    _boxes_in         = boxes_in;
    _batch_splits_in  = batch_splits_in;
    _scores_out       = scores_out;
    _boxes_out        = boxes_out;
    _classes          = classes;
    _batch_splits_out = batch_splits_out;
    _keeps            = keeps;
    _keeps_size       = keeps_size;
    _info             = info;

    // Configure kernel window
    Window win = calculate_max_window(*scores_in->info(), Steps(scores_in->info()->dimension(0)));

    IKernel::configure(win);
}

void CPPBoxWithNonMaximaSuppressionLimitKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IKernel::window(), window);

    switch(_scores_in->info()->data_type())
    {
        case DataType::F32:
            run_nmslimit<float>();
            break;
        case DataType::F16:
            run_nmslimit<half>();
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
} // namespace arm_compute
