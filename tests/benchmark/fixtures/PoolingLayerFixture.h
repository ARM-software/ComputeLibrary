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
#ifndef ARM_COMPUTE_TEST_POOLINGLAYERFIXTURE
#define ARM_COMPUTE_TEST_POOLINGLAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
using namespace arm_compute::misc::shape_calculator;

/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class PoolingLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, PoolingLayerInfo info, DataType data_type, DataLayout data_layout, int batches)
    {
        // Set batched in source and destination shapes

        // Permute shape if NHWC format
        if(data_layout == DataLayout::NHWC)
        {
            permute(src_shape, PermutationVector(2U, 0U, 1U));
        }

        TensorInfo src_info(src_shape, 1, data_type);
        src_info.set_data_layout(data_layout);

        TensorShape dst_shape = compute_pool_shape(src_info, info);

        src_shape.set(src_shape.num_dimensions(), batches);
        dst_shape.set(dst_shape.num_dimensions(), batches);

        // Create tensors
        src = create_tensor<TensorType>(src_shape, data_type, 1, QuantizationInfo(), data_layout);
        dst = create_tensor<TensorType>(dst_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        pool_layer.configure(&src, &dst, info);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        pool_layer.run();
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
    TensorType src{};
    TensorType dst{};
    Function   pool_layer{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLINGLAYERFIXTURE */
