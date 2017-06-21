/*
 * Copyright (c) 2017 ARM Limited.
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
#include "validation/Helpers.h"

using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
namespace validation
{
std::vector<ROI> generate_random_rois(const TensorShape &shape, const ROIPoolingLayerInfo &pool_info, unsigned int num_rois, std::random_device::result_type seed)
{
    ARM_COMPUTE_ERROR_ON((pool_info.pooled_width() < 4) || (pool_info.pooled_height() < 4));

    std::vector<ROI> rois;
    std::mt19937     gen(seed);
    const int        pool_width  = pool_info.pooled_width();
    const int        pool_height = pool_info.pooled_height();
    const float      roi_scale   = pool_info.spatial_scale();

    // Calculate distribution bounds
    const auto scaled_width  = static_cast<int>((shape.x() / roi_scale) / pool_width);
    const auto scaled_height = static_cast<int>((shape.y() / roi_scale) / pool_height);
    const auto min_width     = static_cast<int>(pool_width / roi_scale);
    const auto min_height    = static_cast<int>(pool_height / roi_scale);

    // Create distributions
    std::uniform_int_distribution<int> dist_batch(0, shape[3] - 1);
    std::uniform_int_distribution<int> dist_x(0, scaled_width);
    std::uniform_int_distribution<int> dist_y(0, scaled_height);
    std::uniform_int_distribution<int> dist_w(min_width, std::max(min_width, (pool_width - 2) * scaled_width));
    std::uniform_int_distribution<int> dist_h(min_height, std::max(min_height, (pool_height - 2) * scaled_height));

    for(unsigned int r = 0; r < num_rois; ++r)
    {
        ROI roi{};
        roi.batch_idx   = dist_batch(gen);
        roi.rect.x      = dist_x(gen);
        roi.rect.y      = dist_y(gen);
        roi.rect.width  = dist_w(gen);
        roi.rect.height = dist_h(gen);
        rois.push_back(roi);
    }

    return rois;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
