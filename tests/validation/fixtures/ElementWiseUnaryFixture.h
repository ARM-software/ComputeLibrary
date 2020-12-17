/*
 * Copyright (c) 2018-2021 Arm Limited.
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
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementWiseUnaryValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType input_data_type, bool in_place, ElementWiseUnary op)
    {
        _op        = op;
        _target    = compute_target(input_shape, input_data_type, in_place);
        _reference = compute_reference(input_shape, input_data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType data_type)
    {
        using FloatType             = typename std::conditional < std::is_same<T, half>::value || std::is_floating_point<T>::value, T, float >::type;
        using FloatDistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_fp16, std::uniform_real_distribution<FloatType>>::type;

        switch(_op)
        {
            case ElementWiseUnary::EXP:
            {
                FloatDistributionType distribution{ FloatType(-1.0f), FloatType(1.0f) };
                library->fill(tensor, distribution, i);
                break;
            }
            case ElementWiseUnary::RSQRT:
            {
                FloatDistributionType distribution{ FloatType(1.0f), FloatType(2.0f) };
                library->fill(tensor, distribution, i);
                break;
            }
            case ElementWiseUnary::ABS:
            case ElementWiseUnary::NEG:
            {
                switch(data_type)
                {
                    case DataType::F16:
                    {
                        arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -2.0f, 2.0f };
                        library->fill(tensor, distribution, i);
                        break;
                    }
                    case DataType::F32:
                    {
                        FloatDistributionType distribution{ FloatType(-2.0f), FloatType(2.0f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }
                    case DataType::S32:
                    {
                        std::uniform_int_distribution<int32_t> distribution(-100, 100);
                        library->fill(tensor, distribution, i);
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("DataType for Elementwise Negation Not implemented");
                }
                break;
            }
            case ElementWiseUnary::LOG:
            {
                FloatDistributionType distribution{ FloatType(0.0000001f), FloatType(100.0f) };
                library->fill(tensor, distribution, i);
                break;
            }
            case ElementWiseUnary::SIN:
            {
                FloatDistributionType distribution{ FloatType(-100.00f), FloatType(100.00f) };
                library->fill(tensor, distribution, i);
                break;
            }
            case ElementWiseUnary::ROUND:
            {
                FloatDistributionType distribution{ FloatType(100.0f), FloatType(-100.0f) };
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, bool in_place)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        TensorType *actual_dst = in_place ? &src : &dst;

        // Create and configure function
        FunctionType elwiseunary_layer;
        elwiseunary_layer.configure(&src, actual_dst);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        if(!in_place)
        {
            ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
            dst.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src), 0, data_type);

        // Compute function
        elwiseunary_layer.run();

        if(in_place)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src, 0, data_type);

        return reference::elementwise_unary<T>(src, _op);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    ElementWiseUnary _op{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RsqrtValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::RSQRT);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ExpValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::EXP);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::NEG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegValidationInPlaceFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type, bool in_place)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, in_place, ElementWiseUnary::NEG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::LOG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AbsValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ABS);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SinValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::SIN);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RoundValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ROUND);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ELEMENTWISE_UNARY_FIXTURE */
