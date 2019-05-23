/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE
#define ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationGenericValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &shape0,
               const TensorShape &shape1,
               DataType           dt_in1,
               DataType           dt_in2,
               float              scale,
               ConvertPolicy      convert_policy,
               RoundingPolicy     rounding_policy,
               QuantizationInfo   qinfo0,
               QuantizationInfo   qinfo1,
               QuantizationInfo   qinfo_out)
    {
        _target    = compute_target(shape0, shape1, dt_in1, dt_in2, scale, convert_policy, rounding_policy, qinfo0, qinfo1, qinfo_out);
        _reference = compute_reference(shape0, shape1, dt_in1, dt_in2, scale, convert_policy, rounding_policy, qinfo0, qinfo1, qinfo_out);
    }

protected:
    template <typename U>
    void fill(U &&tensor, unsigned int seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, DataType dt_in1, DataType dt_in2,
                              float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                              QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create tensors
        TensorType src1 = create_tensor<TensorType>(shape0, dt_in1, 1, qinfo0);
        TensorType src2 = create_tensor<TensorType>(shape1, dt_in2, 1, qinfo1);
        TensorType dst  = create_tensor<TensorType>(TensorShape::broadcast_shape(shape0, shape1), dt_in2, 1, qinfo_out);

        // Create and configure function
        FunctionType multiply;
        multiply.configure(&src1, &src2, &dst, scale, convert_policy, rounding_policy);

        ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src1), 0);
        fill(AccessorType(src2), 1);

        // Compute function
        multiply.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape0, const TensorShape &shape1, DataType dt_in1, DataType dt_in2,
                                       float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
                                       QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create reference
        SimpleTensor<T1> src1{ shape0, dt_in1, 1, qinfo0 };
        SimpleTensor<T2> src2{ shape1, dt_in2, 1, qinfo1 };

        // Fill reference
        fill(src1, 0);
        fill(src2, 1);

        return reference::pixel_wise_multiplication<T1, T2>(src1, src2, scale, convert_policy, rounding_policy, qinfo_out);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationQuatizedValidationFixture : public PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType dt_in1, DataType dt_in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
               QuantizationInfo in1_qua_info, QuantizationInfo in2_qua_info, QuantizationInfo out_qua_info)
    {
        PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, shape, dt_in1, dt_in2, scale, convert_policy, rounding_policy,
                                                                                                               in1_qua_info, in2_qua_info, out_qua_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationValidationFixture : public PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType dt_in1, DataType dt_in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
    {
        PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, shape, dt_in1, dt_in2, scale, convert_policy, rounding_policy,
                                                                                                               QuantizationInfo(), QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationBroadcastValidationFixture : public PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType dt_in1, DataType dt_in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
    {
        PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape0, shape1, dt_in1, dt_in2, scale, convert_policy, rounding_policy,
                                                                                                               QuantizationInfo(), QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationValidationQuantizedFixture : public PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType dt, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        PixelWiseMultiplicationGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, shape, dt, dt, scale, convert_policy, rounding_policy,
                                                                                                               qinfo0, qinfo1, qinfo_out);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE */
