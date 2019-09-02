/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_ELEMENTWISE_UNARY_FIXTURE
#define ARM_COMPUTE_TEST_ELEMENTWISE_UNARY_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ElementWiseUnary.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementWiseUnaryBenchmarkFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType input_data_type, ElementWiseUnary op)
    {
        src = create_tensor<TensorType>(input_shape, input_data_type);
        dst = create_tensor<TensorType>(input_shape, input_data_type);

        elwiseunary_layer.configure(&src, &dst);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        elwiseunary_layer.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

private:
    TensorType   src{};
    TensorType   dst{};
    FunctionType elwiseunary_layer{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RsqrtBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::RSQRT);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ExpBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::EXP);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::NEG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::LOG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AbsBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::ABS);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SinBenchmarkFixture : public ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryBenchmarkFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, ElementWiseUnary::SIN);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ELEMENTWISE_UNARY_FIXTURE */
