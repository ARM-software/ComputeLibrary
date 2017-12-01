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
#include "tests/validation/reference/DequantizationLayer.h"

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
        _target    = compute_target(shape, data_type);
        _reference = compute_reference(shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    template <typename U>
    void fill_min_max(U &&tensor)
    {
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

        Window window;

        window.set(0, Window::Dimension(0, tensor.shape()[0], 2));

        for(unsigned int d = 1; d < tensor.shape().num_dimensions(); ++d)
        {
            window.set(d, Window::Dimension(0, tensor.shape()[d], 1));
        }

        execute_window_loop(window, [&](const Coordinates & id)
        {
            const float n1 = distribution(gen);
            const float n2 = distribution(gen);

            float min = 0.0f;
            float max = 0.0f;

            if(n1 < n2)
            {
                min = n1;
                max = n2;
            }
            else
            {
                min = n2;
                max = n1;
            }

            auto out_ptr = reinterpret_cast<float *>(tensor(id));
            out_ptr[0]   = min;
            out_ptr[1]   = max;
        });
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type)
    {
        TensorShape shape_min_max = shape;
        shape_min_max.set(Window::DimX, 2);

        // Remove Y and Z dimensions and keep the batches
        shape_min_max.remove_dimension(1);
        shape_min_max.remove_dimension(1);

        // Create tensors
        TensorType src     = create_tensor<TensorType>(shape, data_type);
        TensorType dst     = create_tensor<TensorType>(shape, DataType::F32);
        TensorType min_max = create_tensor<TensorType>(shape_min_max, DataType::F32);

        // Create and configure function
        FunctionType dequantization_layer;
        dequantization_layer.configure(&src, &dst, &min_max);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(min_max.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        min_max.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!min_max.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));
        fill_min_max(AccessorType(min_max));

        // Compute function
        dequantization_layer.run();

        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape &shape, DataType data_type)
    {
        TensorShape shape_min_max = shape;
        shape_min_max.set(Window::DimX, 2);

        // Remove Y and Z dimensions and keep the batches
        shape_min_max.remove_dimension(1);
        shape_min_max.remove_dimension(1);

        // Create reference
        SimpleTensor<T>     src{ shape, data_type };
        SimpleTensor<float> min_max{ shape_min_max, data_type };

        // Fill reference
        fill(src);
        fill_min_max(min_max);

        return reference::dequantization_layer<T>(src, min_max);
    }

    TensorType          _target{};
    SimpleTensor<float> _reference{};
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
