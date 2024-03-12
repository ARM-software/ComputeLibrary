/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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

#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ElementwiseUnary.h"

#include <tuple>
#include <limits>
#include <type_traits>
#include <vector>

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
    void setup(TensorShape input_shape, DataType input_data_type, bool in_place, ElementWiseUnary op,
               bool use_dynamic_shape = false, QuantizationInfo qinfo = QuantizationInfo(), QuantizationInfo qinfo_out = QuantizationInfo())
    {
        _op                = op;
        _target            = compute_target(input_shape, input_data_type, in_place, qinfo, qinfo_out);
        _reference         = compute_reference(input_shape, input_data_type, qinfo, qinfo_out);
        _use_dynamic_shape = use_dynamic_shape;
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType data_type)
    {
        using FloatType             = typename std::conditional < std::is_same<T, half>::value || std::is_floating_point<T>::value, T, float >::type;
        using FloatDistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<FloatType>>::type;

        switch(_op)
        {
            case ElementWiseUnary::EXP:
            {
                switch(data_type)
                {
                    case DataType::F32:
                    {
                        FloatDistributionType distribution{ FloatType(-86.63f), FloatType(88.36f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::F16:
                    {
                        FloatDistributionType distribution{ FloatType(-9.00f), FloatType(10.73f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::QASYMM8:
                    case DataType::QASYMM8_SIGNED:
                        library->fill_tensor_uniform(tensor, i);
                        break;

                    default:
                        ARM_COMPUTE_ERROR("Not implemented");
                }

                break;
            }
            case ElementWiseUnary::RSQRT:
            case ElementWiseUnary::LOG:
            {
                // For floating-point data type, the chosen input range is all positive numbers
                // (i.e. positive and negative zeros are excluded).
                switch(data_type)
                {
                    case DataType::F32:
                    {
                        FloatDistributionType distribution{ std::numeric_limits<float>::min(), std::numeric_limits<float>::max() };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::F16:
                    {
                        FloatDistributionType distribution{ FloatType(0.00006103515625f), FloatType(65504.0f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::QASYMM8:
                    case DataType::QASYMM8_SIGNED:
                        library->fill_tensor_uniform(tensor, i);
                        break;

                    default:
                        ARM_COMPUTE_ERROR("Not implemented");
                }

                break;
            }
            case ElementWiseUnary::SIN:
            {
                switch(data_type)
                {
                    case DataType::F32:
                    case DataType::F16:
                    {
                        FloatDistributionType distribution{ FloatType(-100.0f), FloatType(100.0f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::S32:
                    {
                        std::uniform_int_distribution<int32_t> distribution(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::QASYMM8:
                    case DataType::QASYMM8_SIGNED:
                        library->fill_tensor_uniform(tensor, i);
                        break;

                    default:
                        ARM_COMPUTE_ERROR("Not implemented");
                }

                break;
            }
            case ElementWiseUnary::ABS:
            case ElementWiseUnary::NEG:
            case ElementWiseUnary::ROUND:
            {
                switch(data_type)
                {
                    case DataType::F32:
                    {
                        FloatDistributionType distribution{ std::numeric_limits<float>::lowest() / 2, std::numeric_limits<float>::max() / 2 };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::F16:
                    {
                        FloatDistributionType distribution{ FloatType(-65504.0f), FloatType(65504.0f) };
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::S32:
                    {
                        std::uniform_int_distribution<int32_t> distribution(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
                        library->fill(tensor, distribution, i);
                        break;
                    }

                    case DataType::QASYMM8:
                    case DataType::QASYMM8_SIGNED:
                        library->fill_tensor_uniform(tensor, i);
                        break;

                    default:
                        ARM_COMPUTE_ERROR("Not implemented");
                }

                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, bool in_place, QuantizationInfo qinfo, QuantizationInfo qinfo_out)
    {
        // Create tensors
        TensorType  src        = create_tensor<TensorType>(shape, data_type, 1, qinfo);
        TensorType  dst        = create_tensor<TensorType>(shape, data_type, 1, qinfo_out);
        TensorType *actual_dst = in_place ? &src : &dst;

        // if _use_dynamic_shape is true, this fixture will test scenario for dynamic shapes.
        // - At configure time, all input tensors are marked as dynamic using set_tensor_dynamic()
        // - After configure, tensors are marked as static for run using set_tensor_static()
        // - The tensors with static shape are given to run()
        if(_use_dynamic_shape)
        {
            set_tensor_dynamic(src);
        }

        // Create and configure function
        FunctionType elwiseunary_layer;
        elwiseunary_layer.configure(&src, actual_dst);

        if(_use_dynamic_shape)
        {
            set_tensor_static(src);
        }

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        src.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        if(!in_place)
        {
            ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
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

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, QuantizationInfo qinfo, QuantizationInfo qinfo_out)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, qinfo };
        SimpleTensor<T> dst{ shape, data_type, 1, qinfo_out };

        // Fill reference
        fill(src, 0, data_type);

        return reference::elementwise_unary<T>(src, dst, _op);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    ElementWiseUnary _op{};
    bool             _use_dynamic_shape{ false };
    QuantizationInfo _input_qinfo{};
    QuantizationInfo _output_qinfo{};
};
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RsqrtQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo qinfo, QuantizationInfo qinfo_out)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::RSQRT, false, qinfo, qinfo_out);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RsqrtValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::RSQRT);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RsqrtDynamicShapeValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::RSQRT, true);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ExpValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::EXP);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ExpQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::EXP, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::NEG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::NEG, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegValidationInPlaceFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, bool in_place)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, in_place, ElementWiseUnary::NEG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NegQuantizedValidationInPlaceFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, bool in_place, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, in_place, ElementWiseUnary::NEG, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::LOG);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::LOG, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AbsValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ABS);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AbsQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ABS, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SinValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::SIN);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SinQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::SIN, false, iq, oq);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RoundValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ROUND);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RoundQuantizedValidationFixture : public ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo iq, QuantizationInfo oq)
    {
        ElementWiseUnaryValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, false, ElementWiseUnary::ROUND, false, iq, oq);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ELEMENTWISE_UNARY_FIXTURE */
