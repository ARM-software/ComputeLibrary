/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_REMAP_FIXTURE
#define ARM_COMPUTE_TEST_REMAP_FIXTURE

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
template <typename TensorType, typename Function, typename Accessor>
class RemapFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &input_shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        const uint8_t                          constant_border_value = distribution(gen);

        // Create tensors
        src   = create_tensor<TensorType>(input_shape, data_type);
        map_x = create_tensor<TensorType>(input_shape, DataType::F32);
        map_y = create_tensor<TensorType>(input_shape, DataType::F32);
        dst   = create_tensor<TensorType>(input_shape, data_type);

        // Create and configure function
        remap_func.configure(&src, &map_x, &map_y, &dst, policy, border_mode, constant_border_value);

        // Allocate tensors
        src.allocator()->allocate();
        map_x.allocator()->allocate();
        map_y.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill tensors
        fill(Accessor(src), 0, 0, 255);
        fill(Accessor(map_x), 1, -5, input_shape.x() + 5);
        fill(Accessor(map_y), 2, -5, input_shape.y() + 5);
    }

    void run()
    {
        remap_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        map_x.allocator()->free();
        map_y.allocator()->free();
        dst.allocator()->free();
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        std::uniform_int_distribution<> distribution((int)min, (int)max);
        library->fill(tensor, distribution, i);
    }

private:
    TensorType src{};
    TensorType map_x{};
    TensorType map_y{};
    TensorType dst{};
    Function   remap_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REMAP_FIXTURE */
