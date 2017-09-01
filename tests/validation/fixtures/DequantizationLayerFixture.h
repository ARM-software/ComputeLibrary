/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/CPP/DequantizationLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DequantizationValidationFixedPointFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        // Initialize random min and max values
        rand_min_max(&_min, &_max);

        _target    = compute_target(shape, data_type, _min, _max);
        _reference = compute_reference(shape, data_type, _min, _max);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, float min, float max)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, DataType::F32);

        // Create and configure function
        FunctionType dequantization_layer;
        dequantization_layer.configure(&src, &dst, &min, &max);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        dequantization_layer.run();

        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape &shape, DataType data_type, float min, float max)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        return reference::dequantization_layer<T>(src, min, max);
    }

    /** Generate random constant values to be used as min and max for dequantization.
     */
    void rand_min_max(float *min, float *max)
    {
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> distribution(-10000.0, 10000.0);

        const float n1 = distribution(gen);
        const float n2 = distribution(gen);

        if(n1 < n2)
        {
            *min = n1;
            *max = n2;
        }
        else
        {
            *min = n2;
            *max = n1;
        }
    }

    TensorType          _target{};
    SimpleTensor<float> _reference{};
    float               _min = 0.f;
    float               _max = 0.f;
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DequantizationValidationFixture : public DequantizationValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        DequantizationValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEQUANTIZATION_LAYER_FIXTURE */
