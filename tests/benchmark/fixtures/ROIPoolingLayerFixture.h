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
#ifndef ARM_COMPUTE_TEST_ROIPOOLINGLAYERFIXTURE
#define ARM_COMPUTE_TEST_ROIPOOLINGLAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename AccessorType, typename T>
class ROIPoolingLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const ROIPoolingLayerInfo pool_info, TensorShape rois_shape, DataType data_type, int batches)
    {
        // Set batched in source and destination shapes

        TensorShape shape_dst;
        rois_tensor = create_tensor<TensorType>(rois_shape, DataType::U16);

        input_shape.set(input_shape.num_dimensions(), batches);
        shape_dst.set(0, pool_info.pooled_width());
        shape_dst.set(1, pool_info.pooled_height());
        shape_dst.set(2, input_shape.z());
        shape_dst.set(3, rois_shape[1]);

        // Create tensors
        src = create_tensor<TensorType>(input_shape, data_type, 1);
        dst = create_tensor<TensorType>(shape_dst, data_type, 1);

        // Create and configure function
        roi_pool.configure(&src, &rois_tensor, &dst, pool_info);

        // Allocate tensors
        rois_tensor.allocator()->allocate();
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Create random ROIs
        generate_rois(AccessorType(rois_tensor), input_shape, pool_info, rois_shape);
    }

    void run()
    {
        roi_pool.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
    }

protected:
    template <typename U>
    void generate_rois(U &&rois, const TensorShape &shape, const ROIPoolingLayerInfo &pool_info, TensorShape rois_shape)
    {
        const size_t values_per_roi = rois_shape.x();
        const size_t num_rois       = rois_shape.y();

        std::mt19937 gen(library->seed());
        uint16_t    *rois_ptr = static_cast<uint16_t *>(rois.data());

        const float pool_width  = pool_info.pooled_width();
        const float pool_height = pool_info.pooled_height();
        const float roi_scale   = pool_info.spatial_scale();

        // Calculate distribution bounds
        const auto scaled_width  = static_cast<uint16_t>((shape.x() / roi_scale) / pool_width);
        const auto scaled_height = static_cast<uint16_t>((shape.y() / roi_scale) / pool_height);
        const auto min_width     = static_cast<uint16_t>(pool_width / roi_scale);
        const auto min_height    = static_cast<uint16_t>(pool_height / roi_scale);

        // Create distributions
        std::uniform_int_distribution<int>      dist_batch(0, shape[3] - 1);
        std::uniform_int_distribution<uint16_t> dist_x1(0, scaled_width);
        std::uniform_int_distribution<uint16_t> dist_y1(0, scaled_height);
        std::uniform_int_distribution<uint16_t> dist_w(min_width, std::max(float(min_width), (pool_width - 2) * scaled_width));
        std::uniform_int_distribution<uint16_t> dist_h(min_height, std::max(float(min_height), (pool_height - 2) * scaled_height));

        for(unsigned int pw = 0; pw < num_rois; ++pw)
        {
            const auto batch_idx = dist_batch(gen);
            const auto x1        = dist_x1(gen);
            const auto y1        = dist_y1(gen);
            const auto x2        = x1 + dist_w(gen);
            const auto y2        = y1 + dist_h(gen);

            rois_ptr[values_per_roi * pw]     = batch_idx;
            rois_ptr[values_per_roi * pw + 1] = x1;
            rois_ptr[values_per_roi * pw + 2] = y1;
            rois_ptr[values_per_roi * pw + 3] = x2;
            rois_ptr[values_per_roi * pw + 4] = y2;
        }
    }

private:
    TensorType src{};
    TensorType dst{};
    TensorType rois_tensor{};
    Function   roi_pool{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ROIPOOLINGLAYERFIXTURE */
