/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_ACTIVATIONLAYERFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_ACTIVATIONLAYERFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/ParametersLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/helpers/ActivationHelpers.h"
#include "tests/validation/reference/ActivationLayer.h"

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
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type,
               QuantizationInfo                        quantization_info,
               bool                                    padding_after_configure)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        ActivationLayerInfo info(function, alpha_beta, alpha_beta);

        _in_place                 = in_place;
        _data_type                = data_type;
        _output_quantization_info = helper::calculate_output_quantization_info(_data_type, info, quantization_info);
        _input_quantization_info  = in_place ? _output_quantization_info : quantization_info;
        _padding_after_configure  = padding_after_configure;

        _function  = function;
        _target    = compute_target(shape, info);
        _reference = compute_reference(shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if (is_data_type_float(_data_type))
        {
            float min_bound                = 0;
            float max_bound                = 0;
            std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<T>(_function, _data_type);
            library->fill_static_values(
                tensor, helper::get_boundary_values(_data_type, static_cast<T>(min_bound), static_cast<T>(max_bound)));
        }
        else
        {
            PixelValue min{};
            PixelValue max{};
            std::tie(min, max) = get_min_max(tensor.data_type());
            library->fill_static_values(tensor, helper::get_boundary_values(_data_type, min.get<T>(), max.get<T>()));
        }
    }

    TensorType compute_target(const TensorShape &shape, ActivationLayerInfo info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, _data_type, 1, _input_quantization_info, DataLayout::NCHW);
        TensorType dst = create_tensor<TensorType>(shape, _data_type, 1, _output_quantization_info, DataLayout::NCHW);

        // Create and configure function
        FunctionType act_layer;

        TensorType *dst_ptr = _in_place ? nullptr : &dst;

        act_layer.configure(&src, dst_ptr, info);

        if (_padding_after_configure)
        {
            add_padding_x({&src, &dst});
        }

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());

        if (!_in_place)
        {
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        act_layer.run();

        if (_in_place)
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
        SimpleTensor<T> src{shape, _data_type, 1, _input_quantization_info};

        // Fill reference
        fill(src);

        return reference::activation_layer<T>(src, info, _output_quantization_info);
    }

protected:
    TensorType                              _target{};
    SimpleTensor<T>                         _reference{};
    bool                                    _in_place{};
    bool                                    _padding_after_configure{};
    QuantizationInfo                        _input_quantization_info{};
    QuantizationInfo                        _output_quantization_info{};
    DataType                                _data_type{};
    ActivationLayerInfo::ActivationFunction _function{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationValidationFixture : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, QuantizationInfo(), false /* padding_after_configure */);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationWithPaddingValidationFixture
    : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, QuantizationInfo(), true /* padding_after_configure */);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationValidationQuantizedFixture
    : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type,
               QuantizationInfo                        quantization_info)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, quantization_info, false /* padding_after_configure */);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ActivationWithPaddingValidationQuantizedFixture
    : public ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type,
               QuantizationInfo                        quantization_info)
    {
        ActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, quantization_info, true /* padding_after_configure */);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_ACTIVATIONLAYERFIXTURE_H
