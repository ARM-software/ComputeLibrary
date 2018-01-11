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
#ifndef ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class DequantizationLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type_src, DataType data_type_dst)
    {
        TensorShape shape_min_max = shape;
        shape_min_max.set(Window::DimX, 2);

        // Remove Y and Z dimensions and keep the batches
        shape_min_max.remove_dimension(1);
        shape_min_max.remove_dimension(1);

        // Create tensors
        src     = create_tensor<TensorType>(shape, data_type_src);
        dst     = create_tensor<TensorType>(shape, data_type_dst);
        min_max = create_tensor<TensorType>(shape_min_max, data_type_dst);

        // Create and configure function
        dequantization_func.configure(&src, &dst, &min_max);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        min_max.allocator()->allocate();
    }

    void run()
    {
        dequantization_func.run();
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
        min_max.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    TensorType min_max{};
    Function   dequantization_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE */
