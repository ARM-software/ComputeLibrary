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
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class PoolingLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape dst_shape, PoolingLayerInfo info, DataType data_type, int batches)
    {
        // Set batched in source and destination shapes
        const unsigned int fixed_point_position = 4;
        src_shape.set(src_shape.num_dimensions(), batches);
        dst_shape.set(dst_shape.num_dimensions(), batches);

        // Create tensors
        src = create_tensor<TensorType>(src_shape, data_type, 1, fixed_point_position);
        dst = create_tensor<TensorType>(dst_shape, data_type, 1, fixed_point_position);

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
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLINGLAYERFIXTURE */
