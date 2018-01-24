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
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor, typename Array_T, typename ArrayAccessor>
class ROIPoolingLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, const ROIPoolingLayerInfo pool_info, unsigned int num_rois, DataType data_type, int batches)
    {
        // Set batched in source and destination shapes
        const unsigned int fixed_point_position = 4;
        TensorShape        shape_dst;
        shape.set(shape.num_dimensions(), batches);
        shape_dst.set(0, pool_info.pooled_width());
        shape_dst.set(1, pool_info.pooled_height());
        shape_dst.set(2, shape.z());
        shape_dst.set(3, num_rois);

        // Create tensors
        src = create_tensor<TensorType>(shape, data_type, 1, fixed_point_position);
        dst = create_tensor<TensorType>(shape_dst, data_type, 1, fixed_point_position);

        // Create random ROIs
        std::vector<ROI> rois = generate_random_rois(shape, pool_info, num_rois, 0U);
        rois_array            = arm_compute::support::cpp14::make_unique<Array_T>(num_rois);
        fill_array(ArrayAccessor(*rois_array), rois);

        // Create and configure function
        roi_pool.configure(&src, rois_array.get(), &dst, pool_info);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
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

private:
    TensorType               src{};
    TensorType               dst{};
    std::unique_ptr<Array_T> rois_array{};
    Function                 roi_pool{};
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ROIPOOLINGLAYERFIXTURE */
