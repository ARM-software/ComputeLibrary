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
#include "arm_compute/runtime/CPP/functions/CPPDetectionOutputLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

#include <list>

namespace arm_compute
{
namespace
{
Status detection_layer_validate_arguments(const ITensorInfo *input_loc, const ITensorInfo *input_conf, const ITensorInfo *input_priorbox, const ITensorInfo *output, DetectionOutputLayerInfo info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input_loc, input_conf, input_priorbox, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_loc, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_loc, input_conf, input_priorbox);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_loc->num_dimensions() > 2, "The location input tensor should be [C1, N].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_conf->num_dimensions() > 2, "The location input tensor should be [C2, N].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_priorbox->num_dimensions() > 3, "The priorbox input tensor should be [C3, 2, N].");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.eta() <= 0.f && info.eta() > 1.f, "Eta should be between 0 and 1");

    const int num_priors = input_priorbox->tensor_shape()[0] / 4;
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(static_cast<size_t>((num_priors * info.num_loc_classes() * 4)) != input_loc->tensor_shape()[0], "Number of priors must match number of location predictions.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(static_cast<size_t>((num_priors * info.num_classes())) != input_conf->tensor_shape()[0], "Number of priors must match number of confidence predictions.");

    // Validate configured output
    if(output->total_size() != 0)
    {
        const unsigned int max_size = info.keep_top_k() * (input_loc->num_dimensions() > 1 ? input_loc->dimension(1) : 1);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), TensorShape(7U, max_size));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_loc, output);
    }

    return Status{};
}

/** Function used to sort pair<float, T> in descend order based on the score (first) value.
 */
template <typename T>
bool SortScorePairDescend(const std::pair<float, T> &pair1,
                          const std::pair<float, T> &pair2)
{
    return pair1.first > pair2.first;
}

/** Get location predictions from input_loc.
 *
 * @param[in]  input_loc                The input location prediction.
 * @param[in]  num                      The number of images.
 * @param[in]  num_priors               number of predictions per class.
 * @param[in]  num_loc_classes          number of location classes. It is 1 if share_location is true,
 *                                      and is equal to number of classes needed to predict otherwise.
 * @param[in]  share_location           If true, all classes share the same location prediction.
 * @param[out] all_location_predictions All the location predictions.
 *
 */
void retrieve_all_loc_predictions(const ITensor *input_loc, const int num,
                                  const int num_priors, const int num_loc_classes,
                                  const bool share_location, std::vector<LabelBBox> &all_location_predictions)
{
    for(int i = 0; i < num; ++i)
    {
        for(int c = 0; c < num_loc_classes; ++c)
        {
            int label = share_location ? -1 : c;
            if(all_location_predictions[i].find(label) == all_location_predictions[i].end())
            {
                all_location_predictions[i][label].resize(num_priors);
            }
            else
            {
                ARM_COMPUTE_ERROR_ON(all_location_predictions[i][label].size() != static_cast<size_t>(num_priors));
                break;
            }
        }
    }
    for(int i = 0; i < num; ++i)
    {
        for(int p = 0; p < num_priors; ++p)
        {
            for(int c = 0; c < num_loc_classes; ++c)
            {
                const int label    = share_location ? -1 : c;
                const int base_ptr = i * num_priors * num_loc_classes * 4 + p * num_loc_classes * 4 + c * 4;
                //xmin, ymin, xmax, ymax
                all_location_predictions[i][label][p][0] = *reinterpret_cast<float *>(input_loc->ptr_to_element(Coordinates(base_ptr)));
                all_location_predictions[i][label][p][1] = *reinterpret_cast<float *>(input_loc->ptr_to_element(Coordinates(base_ptr + 1)));
                all_location_predictions[i][label][p][2] = *reinterpret_cast<float *>(input_loc->ptr_to_element(Coordinates(base_ptr + 2)));
                all_location_predictions[i][label][p][3] = *reinterpret_cast<float *>(input_loc->ptr_to_element(Coordinates(base_ptr + 3)));
            }
        }
    }
}

/** Get confidence predictions from input_conf.
 *
 * @param[in]  input_loc                The input location prediction.
 * @param[in]  num                      The number of images.
 * @param[in]  num_priors               Number of predictions per class.
 * @param[in]  num_loc_classes          Number of location classes. It is 1 if share_location is true,
 *                                      and is equal to number of classes needed to predict otherwise.
 * @param[out] all_location_predictions All the location predictions.
 *
 */
void retrieve_all_conf_scores(const ITensor *input_conf, const int num,
                              const int num_priors, const int                 num_classes,
                              std::vector<std::map<int, std::vector<float>>> &all_confidence_scores)
{
    std::vector<float> tmp_buffer;
    tmp_buffer.resize(num * num_priors * num_classes);
    for(int i = 0; i < num; ++i)
    {
        for(int c = 0; c < num_classes; ++c)
        {
            for(int p = 0; p < num_priors; ++p)
            {
                tmp_buffer[i * num_classes * num_priors + c * num_priors + p] =
                    *reinterpret_cast<float *>(input_conf->ptr_to_element(Coordinates(i * num_classes * num_priors + p * num_classes + c)));
            }
        }
    }
    for(int i = 0; i < num; ++i)
    {
        for(int c = 0; c < num_classes; ++c)
        {
            all_confidence_scores[i][c].resize(num_priors);
            all_confidence_scores[i][c].assign(&tmp_buffer[i * num_classes * num_priors + c * num_priors],
                                               &tmp_buffer[i * num_classes * num_priors + c * num_priors + num_priors]);
        }
    }
}

/** Get prior boxes from input_priorbox.
 *
 * @param[in]  input_priorbox           The input location prediction.
 * @param[in]  num_priors               Number of priors.
 * @param[in]  num_loc_classes          number of location classes. It is 1 if share_location is true,
 *                                      and is equal to number of classes needed to predict otherwise.
 * @param[out] all_prior_bboxes         If true, all classes share the same location prediction.
 * @param[out] all_location_predictions All the location predictions.
 *
 */
void retrieve_all_priorbox(const ITensor               *input_priorbox,
                           const int                    num_priors,
                           std::vector<NormalizedBBox> &all_prior_bboxes,
                           std::vector<std::array<float, 4>> &all_prior_variances)
{
    for(int i = 0; i < num_priors; ++i)
    {
        all_prior_bboxes[i] =
        {
            {
                *reinterpret_cast<float *>(input_priorbox->ptr_to_element(Coordinates(i * 4))),
                *reinterpret_cast<float *>(input_priorbox->ptr_to_element(Coordinates(i * 4 + 1))),
                *reinterpret_cast<float *>(input_priorbox->ptr_to_element(Coordinates(i * 4 + 2))),
                *reinterpret_cast<float *>(input_priorbox->ptr_to_element(Coordinates(i * 4 + 3)))
            }
        };
    }

    std::array<float, 4> var({ { 0, 0, 0, 0 } });
    for(int i = 0; i < num_priors; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            var[j] = *reinterpret_cast<float *>(input_priorbox->ptr_to_element(Coordinates((num_priors + i) * 4 + j)));
        }
        all_prior_variances[i] = var;
    }
}

/** Decode a bbox according to a prior bbox.
 *
 * @param[in]  prior_bbox                 The input prior bounding boxes.
 * @param[in]  prior_variance             The corresponding input variance.
 * @param[in]  code_type                  The detection output code type used to decode the results.
 * @param[in]  variance_encoded_in_target If true, the variance is encoded in target.
 * @param[in]  clip_bbox                  If true, the results should be between 0.f and 1.f.
 * @param[in]  bbox                       The input bbox to decode
 * @param[out] decode_bbox                The decoded bboxes.
 *
 */
void DecodeBBox(const NormalizedBBox &prior_bbox, const std::array<float, 4> &prior_variance,
                const DetectionOutputLayerCodeType code_type, const bool variance_encoded_in_target,
                const bool clip_bbox, const NormalizedBBox &bbox, NormalizedBBox &decode_bbox)
{
    // if the variance is encoded in target, we simply need to add the offset predictions
    // otherwise we need to scale the offset accordingly.
    switch(code_type)
    {
        case DetectionOutputLayerCodeType::CORNER:
        {
            decode_bbox[0] = prior_bbox[0] + (variance_encoded_in_target ? bbox[0] : prior_variance[0] * bbox[0]);
            decode_bbox[1] = prior_bbox[1] + (variance_encoded_in_target ? bbox[1] : prior_variance[1] * bbox[1]);
            decode_bbox[2] = prior_bbox[2] + (variance_encoded_in_target ? bbox[2] : prior_variance[2] * bbox[2]);
            decode_bbox[3] = prior_bbox[3] + (variance_encoded_in_target ? bbox[3] : prior_variance[3] * bbox[3]);

            break;
        }
        case DetectionOutputLayerCodeType::CENTER_SIZE:
        {
            const float prior_width  = prior_bbox[2] - prior_bbox[0];
            const float prior_height = prior_bbox[3] - prior_bbox[1];

            // Check if the prior width and height are right
            ARM_COMPUTE_ERROR_ON(prior_width <= 0.f);
            ARM_COMPUTE_ERROR_ON(prior_height <= 0.f);

            const float prior_center_x = (prior_bbox[0] + prior_bbox[2]) / 2.;
            const float prior_center_y = (prior_bbox[1] + prior_bbox[3]) / 2.;

            const float decode_bbox_center_x = (variance_encoded_in_target ? bbox[0] : prior_variance[0] * bbox[0]) * prior_width + prior_center_x;
            const float decode_bbox_center_y = (variance_encoded_in_target ? bbox[1] : prior_variance[1] * bbox[1]) * prior_height + prior_center_y;
            const float decode_bbox_width    = (variance_encoded_in_target ? std::exp(bbox[2]) : std::exp(prior_variance[2] * bbox[2])) * prior_width;
            const float decode_bbox_height   = (variance_encoded_in_target ? std::exp(bbox[3]) : std::exp(prior_variance[3] * bbox[3])) * prior_height;

            decode_bbox[0] = (decode_bbox_center_x - decode_bbox_width / 2.f);
            decode_bbox[1] = (decode_bbox_center_y - decode_bbox_height / 2.f);
            decode_bbox[2] = (decode_bbox_center_x + decode_bbox_width / 2.f);
            decode_bbox[3] = (decode_bbox_center_y + decode_bbox_height / 2.f);

            break;
        }
        case DetectionOutputLayerCodeType::CORNER_SIZE:
        {
            const float prior_width  = prior_bbox[2] - prior_bbox[0];
            const float prior_height = prior_bbox[3] - prior_bbox[1];

            // Check if the prior width and height are greater than 0
            ARM_COMPUTE_ERROR_ON(prior_width <= 0.f);
            ARM_COMPUTE_ERROR_ON(prior_height <= 0.f);

            decode_bbox[0] = prior_bbox[0] + (variance_encoded_in_target ? bbox[0] : prior_variance[0] * bbox[0]) * prior_width;
            decode_bbox[1] = prior_bbox[1] + (variance_encoded_in_target ? bbox[1] : prior_variance[1] * bbox[1]) * prior_height;
            decode_bbox[2] = prior_bbox[2] + (variance_encoded_in_target ? bbox[2] : prior_variance[2] * bbox[2]) * prior_width;
            decode_bbox[3] = prior_bbox[3] + (variance_encoded_in_target ? bbox[3] : prior_variance[3] * bbox[3]) * prior_height;

            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported Detection Output Code Type.");
    }

    if(clip_bbox)
    {
        for(auto &d_bbox : decode_bbox)
        {
            d_bbox = utility::clamp(d_bbox, 0.f, 1.f);
        }
    }
}

/** Do non maximum suppression given bboxes and scores.
 *
 * @param[in]  bboxes          The input bounding boxes.
 * @param[in]  scores          The corresponding input confidence.
 * @param[in]  score_threshold The threshold used to filter detection results.
 * @param[in]  nms_threshold   The threshold used in non maximum suppression.
 * @param[in]  eta             Adaptation rate for nms threshold.
 * @param[in]  top_k           If not -1, keep at most top_k picked indices.
 * @param[out] indices         The kept indices of bboxes after nms.
 *
 */
void ApplyNMSFast(const std::vector<NormalizedBBox> &bboxes,
                  const std::vector<float> &scores, const float score_threshold,
                  const float nms_threshold, const float eta, const int top_k,
                  std::vector<int> &indices)
{
    ARM_COMPUTE_ERROR_ON_MSG(bboxes.size() != scores.size(), "bboxes and scores have different size.");

    // Get top_k scores (with corresponding indices).
    std::list<std::pair<float, int>> score_index_vec;

    // Generate index score pairs.
    for(size_t i = 0; i < scores.size(); ++i)
    {
        if(scores[i] > score_threshold)
        {
            score_index_vec.emplace_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    score_index_vec.sort(SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    const int score_index_vec_size = score_index_vec.size();
    if(top_k > -1 && top_k < score_index_vec_size)
    {
        score_index_vec.resize(top_k);
    }

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();

    while(!score_index_vec.empty())
    {
        const int idx  = score_index_vec.front().second;
        bool      keep = true;
        for(int kept_idx : indices)
        {
            if(keep)
            {
                // Compute the jaccard (intersection over union IoU) overlap between two bboxes.
                NormalizedBBox intersect_bbox = std::array<float, 4>({ { 0, 0, 0, 0 } });
                if(bboxes[kept_idx][0] > bboxes[idx][2] || bboxes[kept_idx][2] < bboxes[idx][0] || bboxes[kept_idx][1] > bboxes[idx][3] || bboxes[kept_idx][3] < bboxes[idx][1])
                {
                    intersect_bbox = std::array<float, 4>({ { 0, 0, 0, 0 } });
                }
                else
                {
                    intersect_bbox = std::array<float, 4>({ {
                            std::max(bboxes[idx][0], bboxes[kept_idx][0]),
                            std::max(bboxes[idx][1], bboxes[kept_idx][1]),
                            std::min(bboxes[idx][2], bboxes[kept_idx][2]),
                            std::min(bboxes[idx][3], bboxes[kept_idx][3])
                        }
                    });
                }

                float intersect_width  = intersect_bbox[2] - intersect_bbox[0];
                float intersect_height = intersect_bbox[3] - intersect_bbox[1];

                float overlap = 0.f;
                if(intersect_width > 0 && intersect_height > 0)
                {
                    float intersect_size = intersect_width * intersect_height;
                    float bbox1_size     = (bboxes[idx][2] < bboxes[idx][0]
                                            || bboxes[idx][3] < bboxes[idx][1]) ?
                                           0.f :
                                           (bboxes[idx][2] - bboxes[idx][0]) * (bboxes[idx][3] - bboxes[idx][1]); //BBoxSize(bboxes[idx]);
                    float bbox2_size = (bboxes[kept_idx][2] < bboxes[kept_idx][0]
                                        || bboxes[kept_idx][3] < bboxes[kept_idx][1]) ?
                                       0.f :
                                       (bboxes[kept_idx][2] - bboxes[kept_idx][0]) * (bboxes[kept_idx][3] - bboxes[kept_idx][1]); // BBoxSize(bboxes[kept_idx]);
                    overlap = intersect_size / (bbox1_size + bbox2_size - intersect_size);
                }
                keep = (overlap <= adaptive_threshold);
            }
            else
            {
                break;
            }
        }
        if(keep)
        {
            indices.push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if(keep && eta < 1.f && adaptive_threshold > 0.5f)
        {
            adaptive_threshold *= eta;
        }
    }
}

Status non_max_suppression_validate_arguments(const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *indices, unsigned int max_output_size,
                                              const float score_threshold, const float nms_threshold)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(bboxes, scores, indices);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bboxes, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scores, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(bboxes->num_dimensions() > 2, "The bboxes tensor must be a 2-D float tensor of shape [4, num_boxes].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(scores->num_dimensions() > 1, "The scores tensor must be a 1-D float tensor of shape [num_boxes].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(indices->num_dimensions() > 1, "The indices must be 1-D integer tensor of shape [M], where max_output_size <= M");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(bboxes, scores);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(scores->num_dimensions() > 1, "Scores must be a 1D float tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(indices->dimension(0) == 0, "Indices tensor must be bigger than 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(max_output_size == 0, "Max size cannot be 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(nms_threshold < 0.f || nms_threshold > 1.f, "Threshould must be in [0,1]");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(score_threshold < 0.f || score_threshold > 1.f, "Threshould must be in [0,1]");

    return Status{};
}
} // namespace

CPPNonMaximumSuppression::CPPNonMaximumSuppression()
    : _bboxes(nullptr), _scores(nullptr), _indices(nullptr), _max_output_size(0), _score_threshold(0.f), _nms_threshold(0.f)
{
}

void CPPNonMaximumSuppression::configure(
    const ITensor *bboxes, const ITensor *scores, ITensor *indices, unsigned int max_output_size,
    const float score_threshold, const float nms_threshold)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(bboxes, scores, indices);
    ARM_COMPUTE_ERROR_THROW_ON(non_max_suppression_validate_arguments(bboxes->info(), scores->info(), indices->info(), max_output_size, score_threshold, nms_threshold));

    // copy scores also to a vector
    _bboxes  = bboxes;
    _scores  = scores;
    _indices = indices;

    _nms_threshold   = nms_threshold;
    _max_output_size = max_output_size;
    _score_threshold = score_threshold;
}

Status CPPNonMaximumSuppression::validate(
    const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *indices, unsigned int max_output_size,
    const float score_threshold, const float nms_threshold)
{
    ARM_COMPUTE_RETURN_ON_ERROR(non_max_suppression_validate_arguments(bboxes, scores, indices, max_output_size, score_threshold, nms_threshold));
    return Status{};
}

void extract_bounding_boxes_from_tensor(const ITensor *bboxes, std::vector<NormalizedBBox> &bboxes_vector)
{
    Window input_win;
    input_win.use_tensor_dimensions(bboxes->info()->tensor_shape());
    input_win.set_dimension_step(0U, 4U);
    input_win.set_dimension_step(1U, 1U);
    Iterator input(bboxes, input_win);
    auto     f = [&bboxes_vector, &input](const Coordinates &)
    {
        const auto input_ptr = reinterpret_cast<const float *>(input.ptr());
        bboxes_vector.push_back(NormalizedBBox({ { *input_ptr, *(input_ptr + 1), *(2 + input_ptr), *(3 + input_ptr) } }));
    };
    execute_window_loop(input_win, f, input);
}

void extract_scores_from_tensor(const ITensor *scores, std::vector<float> &scores_vector)
{
    Window window;
    window.use_tensor_dimensions(scores->info()->tensor_shape());
    Iterator it(scores, window);
    auto     f = [&it, &scores_vector](const Coordinates &)
    {
        const auto input_ptr = reinterpret_cast<const float *>(it.ptr());
        scores_vector.push_back(*input_ptr);
    };
    execute_window_loop(window, f, it);
}

void CPPNonMaximumSuppression::run()
{
    std::vector<NormalizedBBox> bboxes_vector;
    std::vector<float>          scores_vector;
    std::vector<int>            indices_vector;
    extract_bounding_boxes_from_tensor(_bboxes, bboxes_vector);
    extract_scores_from_tensor(_scores, scores_vector);
    ApplyNMSFast(bboxes_vector, scores_vector, _score_threshold, _nms_threshold, 1, -1 /* disable top_k */, indices_vector);
    std::copy_n(indices_vector.begin(), std::min(indices_vector.size(), _indices->info()->dimension(0)), reinterpret_cast<int *>(_indices->ptr_to_element(Coordinates(0))));
}

CPPDetectionOutputLayer::CPPDetectionOutputLayer()
    : _input_loc(nullptr), _input_conf(nullptr), _input_priorbox(nullptr), _output(nullptr), _info(), _num_priors(), _num(), _all_location_predictions(), _all_confidence_scores(), _all_prior_bboxes(),
      _all_prior_variances(), _all_decode_bboxes(), _all_indices()
{
}

void CPPDetectionOutputLayer::configure(const ITensor *input_loc, const ITensor *input_conf, const ITensor *input_priorbox, ITensor *output, DetectionOutputLayerInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_loc, input_conf, input_priorbox, output);
    // Output auto initialization if not yet initialized
    // Since the number of bboxes to kept is unknown before nms, the shape is set to the maximum
    // The maximum is keep_top_k * input_loc_size[1]
    // Each row is a 7 dimension std::vector, which stores [image_id, label, confidence, xmin, ymin, xmax, ymax]
    const unsigned int max_size = info.keep_top_k() * (input_loc->info()->num_dimensions() > 1 ? input_loc->info()->dimension(1) : 1);
    auto_init_if_empty(*output->info(), input_loc->info()->clone()->set_tensor_shape(TensorShape(7U, max_size)));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(detection_layer_validate_arguments(input_loc->info(), input_conf->info(), input_priorbox->info(), output->info(), info));

    _input_loc      = input_loc;
    _input_conf     = input_conf;
    _input_priorbox = input_priorbox;
    _output         = output;
    _info           = info;
    _num_priors     = input_priorbox->info()->dimension(0) / 4;
    _num            = (_input_loc->info()->num_dimensions() > 1 ? _input_loc->info()->dimension(1) : 1);

    _all_location_predictions.resize(_num);
    _all_confidence_scores.resize(_num);
    _all_prior_bboxes.resize(_num_priors);
    _all_prior_variances.resize(_num_priors);
    _all_decode_bboxes.resize(_num);

    for(int i = 0; i < _num; ++i)
    {
        for(int c = 0; c < _info.num_loc_classes(); ++c)
        {
            const int label = _info.share_location() ? -1 : c;
            if(label == _info.background_label_id())
            {
                // Ignore background class.
                continue;
            }
            _all_decode_bboxes[i][label].resize(_num_priors);
        }
    }
    _all_indices.resize(_num);

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
}

Status CPPDetectionOutputLayer::validate(const ITensorInfo *input_loc, const ITensorInfo *input_conf, const ITensorInfo *input_priorbox, const ITensorInfo *output, DetectionOutputLayerInfo info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(detection_layer_validate_arguments(input_loc, input_conf, input_priorbox, output, info));
    return Status{};
}

void CPPDetectionOutputLayer::run()
{
    // Retrieve all location predictions.
    retrieve_all_loc_predictions(_input_loc, _num, _num_priors, _info.num_loc_classes(), _info.share_location(), _all_location_predictions);

    // Retrieve all confidences.
    retrieve_all_conf_scores(_input_conf, _num, _num_priors, _info.num_classes(), _all_confidence_scores);

    // Retrieve all prior bboxes.
    retrieve_all_priorbox(_input_priorbox, _num_priors, _all_prior_bboxes, _all_prior_variances);

    // Decode all loc predictions to bboxes
    const bool clip_bbox = false;
    for(int i = 0; i < _num; ++i)
    {
        for(int c = 0; c < _info.num_loc_classes(); ++c)
        {
            const int label = _info.share_location() ? -1 : c;
            if(label == _info.background_label_id())
            {
                // Ignore background class.
                continue;
            }
            ARM_COMPUTE_ERROR_ON_MSG(_all_location_predictions[i].find(label) == _all_location_predictions[i].end(), "Could not find location predictions for label %d.", label);

            const std::vector<NormalizedBBox> &label_loc_preds = _all_location_predictions[i].find(label)->second;

            const int num_bboxes = _all_prior_bboxes.size();
            ARM_COMPUTE_ERROR_ON(_all_prior_variances[i].size() != 4);

            for(int j = 0; j < num_bboxes; ++j)
            {
                DecodeBBox(_all_prior_bboxes[j], _all_prior_variances[j], _info.code_type(), _info.variance_encoded_in_target(), clip_bbox, label_loc_preds[j], _all_decode_bboxes[i][label][j]);
            }
        }
    }

    int num_kept = 0;

    for(int i = 0; i < _num; ++i)
    {
        const LabelBBox &decode_bboxes = _all_decode_bboxes[i];
        const std::map<int, std::vector<float>> &conf_scores = _all_confidence_scores[i];

        std::map<int, std::vector<int>> indices;
        int num_det = 0;
        for(int c = 0; c < _info.num_classes(); ++c)
        {
            if(c == _info.background_label_id())
            {
                // Ignore background class
                continue;
            }
            const int label = _info.share_location() ? -1 : c;
            if(conf_scores.find(c) == conf_scores.end() || decode_bboxes.find(label) == decode_bboxes.end())
            {
                ARM_COMPUTE_ERROR("Could not find predictions for label %d.", label);
            }
            const std::vector<float>          &scores = conf_scores.find(c)->second;
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;

            ApplyNMSFast(bboxes, scores, _info.confidence_threshold(), _info.nms_threshold(), _info.eta(), _info.top_k(), indices[c]);

            num_det += indices[c].size();
        }

        int num_to_add = 0;
        if(_info.keep_top_k() > -1 && num_det > _info.keep_top_k())
        {
            std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
            for(auto it : indices)
            {
                const int               label         = it.first;
                const std::vector<int> &label_indices = it.second;

                if(conf_scores.find(label) == conf_scores.end())
                {
                    ARM_COMPUTE_ERROR("Could not find predictions for label %d.", label);
                }

                const std::vector<float> &scores = conf_scores.find(label)->second;
                for(auto idx : label_indices)
                {
                    ARM_COMPUTE_ERROR_ON(idx > static_cast<int>(scores.size()));
                    score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }

            // Keep top k results per image.
            std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<std::pair<int, int>>);
            score_index_pairs.resize(_info.keep_top_k());

            // Store the new indices.

            std::map<int, std::vector<int>> new_indices;
            for(auto score_index_pair : score_index_pairs)
            {
                int label = score_index_pair.second.first;
                int idx   = score_index_pair.second.second;
                new_indices[label].push_back(idx);
            }
            _all_indices[i] = new_indices;
            num_to_add      = _info.keep_top_k();
        }
        else
        {
            _all_indices[i] = indices;
            num_to_add      = num_det;
        }
        num_kept += num_to_add;
    }

    //Update the valid region of the ouput to mark the exact number of detection
    _output->info()->set_valid_region(ValidRegion(Coordinates(0, 0), TensorShape(7, num_kept)));

    int count = 0;
    for(int i = 0; i < _num; ++i)
    {
        const std::map<int, std::vector<float>> &conf_scores = _all_confidence_scores[i];
        const LabelBBox &decode_bboxes = _all_decode_bboxes[i];
        for(auto &it : _all_indices[i])
        {
            const int                 label     = it.first;
            const std::vector<float> &scores    = conf_scores.find(label)->second;
            const int                 loc_label = _info.share_location() ? -1 : label;
            if(conf_scores.find(label) == conf_scores.end() || decode_bboxes.find(loc_label) == decode_bboxes.end())
            {
                // Either if there are no confidence predictions
                // or there are no location predictions for current label.
                ARM_COMPUTE_ERROR("Could not find predictions for the label %d.", label);
            }
            const std::vector<NormalizedBBox> &bboxes  = decode_bboxes.find(loc_label)->second;
            const std::vector<int>            &indices = it.second;

            for(auto idx : indices)
            {
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7))))     = i;
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 1)))) = label;
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 2)))) = scores[idx];
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 3)))) = bboxes[idx][0];
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 4)))) = bboxes[idx][1];
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 5)))) = bboxes[idx][2];
                *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(count * 7 + 6)))) = bboxes[idx][3];

                ++count;
            }
        }
    }
}
} // namespace arm_compute
