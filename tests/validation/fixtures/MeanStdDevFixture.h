/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_MEAN_STD_DEV_FIXTURE
#define ARM_COMPUTE_TEST_MEAN_STD_DEV_FIXTURE

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/MeanStdDev.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class MeanStdDevValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        _target    = compute_target(shape, data_type);
        _reference = compute_reference(shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(is_data_type_float(tensor.data_type()))
        {
            std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    std::pair<float, float> compute_target(const TensorShape &shape, DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);

        // Create output variables
        float mean    = 0.0f;
        float std_dev = 0.0f;

        // Create and configure function
        FunctionType mean_std_dev;
        mean_std_dev.configure(&src, &mean, &std_dev);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        mean_std_dev.run();

        return std::make_pair(mean, std_dev);
    }

    std::pair<float, float> compute_reference(const TensorShape &shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        // Compute reference
        return reference::mean_and_standard_deviation<T>(src);
    }

    std::pair<float, float> _target{};
    std::pair<float, float> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MEAN_STD_DEV_FIXTURE */
