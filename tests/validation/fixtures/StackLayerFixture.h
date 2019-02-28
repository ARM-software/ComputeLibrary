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
#ifndef ARM_COMPUTE_TEST_STACK_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_STACK_LAYER_FIXTURE

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/StackLayer.h"
#include "tests/validation/reference/Utils.h"

#include <random>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AbstractTensorType, typename AccessorType, typename FunctionType, typename T>
class StackLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_src, int axis, DataType data_type, int num_tensors)
    {
        _target    = compute_target(shape_src, axis, data_type, num_tensors);
        _reference = compute_reference(shape_src, axis, data_type, num_tensors);
    }

protected:
    template <typename U>
    void fill(U &&tensor, unsigned int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(TensorShape shape_src, int axis, DataType data_type, int num_tensors)
    {
        std::vector<TensorType>           tensors(num_tensors);
        std::vector<AbstractTensorType *> src(num_tensors);

        // Create vector of input tensors
        for(int i = 0; i < num_tensors; ++i)
        {
            tensors[i] = create_tensor<TensorType>(shape_src, data_type);
            src[i]     = &(tensors[i]);
            ARM_COMPUTE_EXPECT(tensors[i].info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Create tensors
        TensorType dst;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        FunctionType stack;
        stack.configure(src, axis, &dst);

        // Allocate and fill the input tensors
        for(int i = 0; i < num_tensors; ++i)
        {
            ARM_COMPUTE_EXPECT(tensors[i].info()->is_resizable(), framework::LogLevel::ERRORS);
            tensors[i].allocator()->allocate();
            ARM_COMPUTE_EXPECT(!tensors[i].info()->is_resizable(), framework::LogLevel::ERRORS);

            // Fill input tensor
            fill(AccessorType(tensors[i]), i);
        }

        // Allocate output tensor
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Compute stack function
        stack.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_src, int axis, DataType data_type, int num_tensors)
    {
        std::vector<SimpleTensor<T>> src;

        for(int i = 0; i < num_tensors; ++i)
        {
            src.emplace_back(std::move(SimpleTensor<T>(shape_src, data_type, 1)));

            fill(src[i], i);
        }

        // Wrap around negative values
        const unsigned int axis_u = wrap_around(axis, static_cast<int>(shape_src.num_dimensions() + 1));

        const TensorShape shape_dst = compute_stack_shape(TensorInfo(shape_src, 1, data_type), axis_u, num_tensors);

        return reference::stack_layer<T>(src, shape_dst, data_type, axis_u);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_STACK_LAYER_FIXTURE */
