/*
 * Copyright (c) 2019 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_FILL_FIXTURE
#define ARM_COMPUTE_TEST_FILL_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FillFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type)
    {
        _target = compute_target(input_shape, data_type);
        _reference = compute_reference(input_shape, data_type);
    }

protected:
    TensorType compute_target(const TensorShape &input_shape, DataType data_type)
    {
        // Create tensors
        TensorType input = create_tensor<TensorType>(input_shape, data_type);

        // Allocate tensors
        input.allocator()->allocate();

        // Fill tensor with an initial value
        library->fill_tensor_uniform(AccessorType(input), 0);

        // Create and configure function
        FunctionType fill;
        const T constant_value {1};
        fill.configure(&input, constant_value);

        // Compute function with a distinct, second value
        fill.run();

        return input;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> input{ input_shape, data_type };

        // Fill tensor
        const T constant_value {1};
        for(int i = 0; i < input.num_elements(); ++i)
        {
            input[i] = constant_value;
        }

        return input;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FILL_FIXTURE */
