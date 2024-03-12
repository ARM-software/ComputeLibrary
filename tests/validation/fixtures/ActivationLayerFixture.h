/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ACTIVATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_ACTIVATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/ParametersLibrary.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationValidationGenericFixture : public framework::Fixture
{
public:
    ActivationValidationGenericFixture()
        : _target(parameters->get_ctx<TensorType>())
    {
    }

    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, DataType data_type, QuantizationInfo quantization_info)
    {
        ActivationLayerInfo info(function, alpha_beta, alpha_beta);

        _in_place                 = in_place;
        _data_type                = data_type;
        _output_quantization_info = calculate_output_quantization_info(_data_type, info, quantization_info);
        _input_quantization_info  = in_place ? _output_quantization_info : quantization_info;

        _function  = function;
        _target    = compute_target(shape, info);
        _reference = compute_reference(shape, info);
    }

protected:
    std::vector<T> get_boundary_values(T min, T max)
    {
        // This function will return a vector filled with the following values that can
        // represent two partitions derived from equivalent partitioning.
        // * Lower parition: min, min + delta, lower quarter (nominal), center - delta
        // * Upper partition: center, center + delta, upper quarter (nominal), max - delta, max
        const auto delta         = is_data_type_float(_data_type) ? T(0.1f) : T(1);
        const auto center_value  = (min + max) / 2;
        const auto lower_quarter = (min + center_value) / 2;
        const auto upper_quarter = (center_value + max) / 2;

        std::vector<T> boundary_values{};

        // To ensure all the inserted values are within the given range after subtracing/adding delta
        auto insert_values = [&boundary_values, &min, &max](const std::initializer_list<T> &new_values)
        {
            for(auto &v : new_values)
            {
                if(v >= min && v <= max)
                {
                    boundary_values.emplace_back(v);
                }
            }
        };

        insert_values({ min, static_cast<T>(min + delta), static_cast<T>(lower_quarter), static_cast<T>(center_value - delta) });                               // lower partition
        insert_values({ static_cast<T>(center_value), static_cast<T>(center_value + delta), static_cast<T>(upper_quarter), static_cast<T>(max - delta), max }); // upper partition

        return boundary_values;
    }

    template <typename U>
    void fill(U &&tensor)
    {
        if(is_data_type_float(_data_type))
        {
            float min_bound = 0;
            float max_bound = 0;
            std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<T>(_function, _data_type);
            library->fill_static_values(tensor, get_boundary_values(static_cast<T>(min_bound), static_cast<T>(max_bound)));
        }
        else
        {
            PixelValue min{};
            PixelValue max{};
            std::tie(min, max) = get_min_max(tensor.data_type());
            library->fill_static_values(tensor, get_boundary_values(min.get<T>(), max.get<T>()));
        }
    }

    TensorType compute_target(const TensorShape &shape, ActivationLayerInfo info)
    {
        auto ctx = parameters->get_ctx<TensorType>();
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, _data_type, 1, _input_quantization_info, DataLayout::NCHW, ctx);
        TensorType dst = create_tensor<TensorType>(shape, _data_type, 1, _output_quantization_info, DataLayout::NCHW, ctx);

        // Create and configure function
        FunctionType act_layer(ctx);

        TensorType *dst_ptr = _in_place ? nullptr : &dst;

        act_layer.configure(&src, dst_ptr, info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());

        if(!_in_place)
        {
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        act_layer.run();

        if(_in_place)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, ActivationLayerInfo info)
    {
        // Create reference
        SimpleTensor<T> src{ shape, _data_type, 1, _input_quantization_info };

        // Fill reference
        fill(src);

        return reference::activation_layer<T>(src, info, _output_quantization_info);
    }

private:
    QuantizationInfo calculate_output_quantization_info(DataType dt, const ActivationLayerInfo &act_info, const QuantizationInfo &default_qinfo)
    {
        auto qasymm8_max        = float(std::numeric_limits<uint8_t>::max()) + 1.f;
        auto qasymm8_signed_max = float(std::numeric_limits<int8_t>::max()) + 1.f;
        auto qsymm16_max        = float(std::numeric_limits<int16_t>::max()) + 1.f;

        switch(act_info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::TANH:
                if(dt == DataType::QSYMM16)
                {
                    return QuantizationInfo(1.f / qsymm16_max, 0);
                }
                else if(dt == DataType::QASYMM8)
                {
                    return QuantizationInfo(1.f / (0.5 * qasymm8_max), int(0.5 * qasymm8_max));
                }
                else if(dt == DataType::QASYMM8_SIGNED)
                {
                    return QuantizationInfo(1.f / qasymm8_signed_max, 0);
                }
                else
                {
                    return default_qinfo;
                }
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                if(dt == DataType::QSYMM16)
                {
                    return QuantizationInfo(1.f / qsymm16_max, 0);
                }
                else if(dt == DataType::QASYMM8)
                {
                    return QuantizationInfo(1.f / qasymm8_max, 0);
                }
                else if(dt == DataType::QASYMM8_SIGNED)
                {
                    return QuantizationInfo(1.f / (2.f * qasymm8_signed_max), -int(qasymm8_signed_max));
                }
                else
                {
                    return default_qinfo;
                }
            default:
                return default_qinfo;
        }
    }

protected:
    TensorType                              _target{};
    SimpleTensor<T>                         _reference{};
    bool                                    _in_place{};
    QuantizationInfo                        _input_quantization_info{};
    QuantizationInfo                        _output_quantization_info{};
    DataType                                _data_type{};
    ActivationLayerInfo::ActivationFunction _function{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationValidationFixture : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, DataType data_type)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, in_place, function, alpha_beta, data_type, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationValidationQuantizedFixture : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, DataType data_type, QuantizationInfo quantization_info)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, in_place, function, alpha_beta, data_type, quantization_info);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ACTIVATION_LAYER_FIXTURE */
