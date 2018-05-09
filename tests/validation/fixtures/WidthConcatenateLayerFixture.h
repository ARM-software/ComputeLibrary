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
#ifndef ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/WidthConcatenateLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename ITensorType, typename AccessorType, typename FunctionType, typename T>
class WidthConcatenateLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        // Create input shapes
        std::mt19937                    gen(library->seed());
        std::uniform_int_distribution<> num_dis(2, 4);
        const int                       num_tensors = num_dis(gen);

        std::vector<TensorShape>         shapes(num_tensors, shape);
        std::bernoulli_distribution      mutate_dis(0.5f);
        std::uniform_real_distribution<> change_dis(-0.25f, 0.f);

        // Generate more shapes based on the input
        for(auto &s : shapes)
        {
            // Randomly change the first dimension
            if(mutate_dis(gen))
            {
                // Decrease the dimension by a small percentage. Don't increase
                // as that could make tensor too large.
                s.set(0, s[0] + 2 * static_cast<int>(s[0] * change_dis(gen)));
            }
        }

        _target    = compute_target(shapes, data_type);
        _reference = compute_reference(shapes, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(std::vector<TensorShape> shapes, DataType data_type)
    {
        std::vector<TensorType>    srcs;
        std::vector<ITensorType *> src_ptrs;

        // Create tensors
        srcs.reserve(shapes.size());

        for(const auto &shape : shapes)
        {
            srcs.emplace_back(create_tensor<TensorType>(shape, data_type, 1, _fractional_bits));
            src_ptrs.emplace_back(&srcs.back());
        }

        TensorShape dst_shape = misc::shape_calculator::calculate_width_concatenate_shape(src_ptrs);
        TensorType  dst       = create_tensor<TensorType>(dst_shape, data_type, 1, _fractional_bits);

        // Create and configure function
        FunctionType width_concat;
        width_concat.configure(src_ptrs, &dst);

        for(auto &src : srcs)
        {
            ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        for(auto &src : srcs)
        {
            src.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        dst.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        int i = 0;
        for(auto &src : srcs)
        {
            fill(AccessorType(src), i++);
        }

        // Compute function
        width_concat.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(std::vector<TensorShape> shapes, DataType data_type)
    {
        std::vector<SimpleTensor<T>> srcs;

        // Create and fill tensors
        int i = 0;
        for(const auto &shape : shapes)
        {
            srcs.emplace_back(shape, data_type, 1, _fractional_bits);
            fill(srcs.back(), i++);
        }

        return reference::widthconcatenate_layer<T>(srcs);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

private:
    int _fractional_bits{ 1 };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WIDTHCONCATENATE_LAYER_FIXTURE */
