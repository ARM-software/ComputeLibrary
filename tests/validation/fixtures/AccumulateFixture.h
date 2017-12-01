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
#ifndef ARM_COMPUTE_TEST_ACCUMULATE_FIXTURE
#define ARM_COMPUTE_TEST_ACCUMULATE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Accumulate.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class AccumulateBaseValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataType output_data_type)
    {
        _target    = compute_target(shape, data_type, output_data_type);
        _reference = compute_reference(shape, data_type, output_data_type);
    }

protected:
    template <typename U, typename D>
    void fill(U &&tensor, int i, D max)
    {
        library->fill_tensor_uniform(tensor, i, static_cast<D>(0), max);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, DataType output_data_type)
    {
        // Create tensors
        TensorType ref_src = create_tensor<TensorType>(shape, data_type);
        TensorType dst     = create_tensor<TensorType>(shape, output_data_type);

        // Create and configure function
        FunctionType accum;
        accum_conf(accum, ref_src, dst);

        ARM_COMPUTE_EXPECT(ref_src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        ref_src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!ref_src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        const T1 max = std::numeric_limits<T1>::max();

        // Fill tensors
        fill(AccessorType(ref_src), 0, max);
        fill(AccessorType(dst), 1, static_cast<T2>(max));

        // Compute function
        accum.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, DataType data_type, DataType output_data_type)
    {
        // Create reference
        SimpleTensor<T1> ref_src{ shape, data_type };

        const T1 max = std::numeric_limits<T1>::max();

        // Fill reference
        fill(ref_src, 0, max);

        return accum_ref(ref_src, output_data_type);
    }

    virtual void accum_conf(FunctionType &func, const TensorType &input, TensorType &accum) = 0;

    virtual SimpleTensor<T2> accum_ref(const SimpleTensor<T1> &input, DataType output_data_type) = 0;

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class AccumulateValidationFixture : public AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataType output_data_type)
    {
        AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, data_type, output_data_type);
    }

    virtual void accum_conf(FunctionType &func, const TensorType &input, TensorType &accum) override
    {
        func.configure(&input, &accum);
    }

    virtual SimpleTensor<T2> accum_ref(const SimpleTensor<T1> &input, DataType output_data_type) override
    {
        return reference::accumulate<T1, T2>(input, output_data_type);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class AccumulateWeightedValidationFixture : public AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataType output_data_type)
    {
        std::mt19937                     gen(library->seed());
        std::uniform_real_distribution<> float_dist(0, 1);

        _alpha = float_dist(gen);

        AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, data_type, output_data_type);
    }

    virtual void accum_conf(FunctionType &func, const TensorType &input, TensorType &accum) override
    {
        func.configure(&input, _alpha, &accum);
    }

    virtual SimpleTensor<T2> accum_ref(const SimpleTensor<T1> &input, DataType output_data_type) override
    {
        return reference::accumulate_weighted<T1, T2>(input, _alpha, output_data_type);
    }

    float _alpha{ 0.f };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class AccumulateSquaredValidationFixture : public AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataType output_data_type)
    {
        std::mt19937                            gen(library->seed());
        std::uniform_int_distribution<uint32_t> int_dist(0, 15);

        _shift = int_dist(gen);

        AccumulateBaseValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, data_type, output_data_type);
    }

    virtual void accum_conf(FunctionType &func, const TensorType &input, TensorType &accum) override
    {
        func.configure(&input, _shift, &accum);
    }

    virtual SimpleTensor<T2> accum_ref(const SimpleTensor<T1> &input, DataType output_data_type) override
    {
        return reference::accumulate_squared<T1, T2>(input, _shift, output_data_type);
    }

    uint32_t _shift{ 0U };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ACCUMULATE_FIXTURE */
