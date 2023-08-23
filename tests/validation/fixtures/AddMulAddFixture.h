/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef TESTS_VALIDATION_FIXTURES_ADDMULADDFIXTURE
#define TESTS_VALIDATION_FIXTURES_ADDMULADDFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"
#include "tests/validation/reference/DequantizationLayer.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"
#include "tests/validation/reference/QuantizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AddMulAddGenericFixture : public framework::Fixture
{
public:
    void setup(const TensorShape &shape, DataType data_type, ActivationLayerInfo &act_info, bool interm_out)
    {
        compute_target(shape, data_type, act_info, interm_out);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType data_type)
    {
        switch(data_type)
        {
            case DataType::F32:
                library->fill_tensor_uniform(tensor, i, -10.f, 10.f);
                break;
            case DataType::F16:
                library->fill_tensor_uniform(tensor, i, -1.f, 1.f);
                break;
            default:
                library->fill_tensor_uniform(tensor, i);
                break;
        }
    }

    void compute_target(const TensorShape &shape, DataType data_type, ActivationLayerInfo &act_info, bool interm_out)
    {
        TensorShape b_shape(shape.x());

        // Create tensors
        TensorType input1       = create_tensor<TensorType>(shape, data_type, 1, _input1_qinfo);
        TensorType input2       = create_tensor<TensorType>(shape, data_type, 1, _input2_qinfo);
        TensorType bn_mul       = create_tensor<TensorType>(b_shape, data_type, 1, _bn_mul_qinfo);
        TensorType bn_add       = create_tensor<TensorType>(b_shape, data_type, 1, _bn_add_qinfo);
        TensorType add_output   = create_tensor<TensorType>(shape, data_type, 1, _add_output_qinfo);
        TensorType final_output = create_tensor<TensorType>(shape, data_type, 1, _final_output_qinfo);

        // Create and configure function
        FunctionType add_mul_add;
        add_mul_add.configure(&input1, &input2, &bn_mul, &bn_add, interm_out ? &add_output : nullptr, &final_output, ConvertPolicy::SATURATE, act_info);

        // Allocate tensors
        input1.allocator()->allocate();
        input2.allocator()->allocate();
        bn_mul.allocator()->allocate();
        bn_add.allocator()->allocate();

        if(interm_out)
        {
            add_output.allocator()->allocate();
        }

        final_output.allocator()->allocate();

        // Fill tensors
        fill(AccessorType(input1), 0, data_type);
        fill(AccessorType(input2), 1, data_type);
        fill(AccessorType(bn_mul), 2, data_type);
        fill(AccessorType(bn_add), 3, data_type);

        // // Compute function
        add_mul_add.run();

        _target = std::move(final_output);

        if(interm_out)
        {
            _interm_target = std::move(add_output);
        }
    }

    TensorType      _target{};
    TensorType      _interm_target{};
    SimpleTensor<T> _reference{};
    SimpleTensor<T> _interm_reference{};

    QuantizationInfo _input1_qinfo{};
    QuantizationInfo _input2_qinfo{};
    QuantizationInfo _bn_mul_qinfo{};
    QuantizationInfo _bn_add_qinfo{};
    QuantizationInfo _add_output_qinfo{};
    QuantizationInfo _final_output_qinfo{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool interm_out>
class AddMulAddFloatValidationFixture : public AddMulAddGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    using Parent = AddMulAddGenericFixture<TensorType, AccessorType, FunctionType, T>;

    void setup(const TensorShape &shape, DataType data_type, ActivationLayerInfo act_info)
    {
        Parent::setup(shape, data_type, act_info, interm_out);
        compute_reference(shape, data_type, act_info);
    }

    // Compute Reference is moved outside of the generic fixture because with the quantized data types,
    // it becomes a very different implementation with intermediate tensors' data types being always float.
    // This way the reference calculations are more readable and the size of the classes will be smaller
    // due to unrepeated fill() and target() methods.
    void compute_reference(const TensorShape &shape, DataType data_type, ActivationLayerInfo &act_info)
    {
        TensorShape b_shape(shape.x());

        // Create reference
        SimpleTensor<T> input1{ shape, data_type };
        SimpleTensor<T> input2{ shape, data_type };
        SimpleTensor<T> bn_mul{ b_shape, data_type };
        SimpleTensor<T> bn_add{ b_shape, data_type };
        SimpleTensor<T> add_output{ shape, data_type, 1 };

        SimpleTensor<T> bn_mul_out{ shape, data_type };
        SimpleTensor<T> bn_add_out{ shape, data_type };

        // Fill reference
        Parent::fill(input1, 0, data_type);
        Parent::fill(input2, 1, data_type);
        Parent::fill(bn_mul, 2, data_type);
        Parent::fill(bn_add, 3, data_type);

        reference::arithmetic_operation<T>(reference::ArithmeticOperation::ADD, input1, input2, add_output, ConvertPolicy::SATURATE);
        bn_mul_out = reference::pixel_wise_multiplication<T, T, T>(add_output, bn_mul, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_UP, data_type);
        reference::arithmetic_operation<T>(reference::ArithmeticOperation::ADD, bn_mul_out, bn_add, bn_add_out, ConvertPolicy::SATURATE);

        if(interm_out)
        {
            Parent::_interm_reference = std::move(add_output);
        }

        if(act_info.enabled() && act_info.activation() != ActivationLayerInfo::ActivationFunction::IDENTITY)
        {
            Parent::_reference = reference::activation_layer(bn_add_out, act_info);
        }
        else
        {
            Parent::_reference = std::move(bn_add_out);
        }
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool interm_out>
class AddMulAddQuantizedValidationFixture : public AddMulAddGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    using Parent = AddMulAddGenericFixture<TensorType, AccessorType, FunctionType, T>;

    void setup(const TensorShape &shape, DataType data_type, ActivationLayerInfo act_info,
               QuantizationInfo input1_qinfo, QuantizationInfo input2_qinfo, QuantizationInfo bn_mul_qinfo,
               QuantizationInfo bn_add_qinfo, QuantizationInfo add_output_qinfo, QuantizationInfo final_output_qinfo)
    {
        // Quantization arguments moved to class attributes to prevent long function declerations
        Parent::_input1_qinfo       = input1_qinfo;
        Parent::_input2_qinfo       = input2_qinfo;
        Parent::_bn_mul_qinfo       = bn_mul_qinfo;
        Parent::_bn_add_qinfo       = bn_add_qinfo;
        Parent::_add_output_qinfo   = add_output_qinfo;
        Parent::_final_output_qinfo = final_output_qinfo;

        Parent::setup(shape, data_type, act_info, interm_out);
        compute_reference(shape, data_type, act_info);
    }

    // Compute Reference is moved outside of the generic fixture because with the quantized data types,
    // it becomes a very different implementation with intermediate tensors' data types being always float.
    // This way the reference calculations are more readable and the size of the classes will be smaller
    // due to unrepeated fill() and target() methods.
    void compute_reference(const TensorShape &shape, DataType data_type, ActivationLayerInfo &act_info)
    {
        TensorShape b_shape(shape.x());

        // Create reference
        SimpleTensor<T> input1{ shape, data_type, 1, Parent::_input1_qinfo };
        SimpleTensor<T> input2{ shape, data_type, 1, Parent::_input2_qinfo };
        SimpleTensor<T> bn_mul{ b_shape, data_type, 1, Parent::_bn_mul_qinfo };
        SimpleTensor<T> bn_add{ b_shape, data_type, 1, Parent::_bn_add_qinfo };

        // Fill input tensors
        Parent::fill(input1, 0, data_type);
        Parent::fill(input2, 1, data_type);
        Parent::fill(bn_mul, 2, data_type);
        Parent::fill(bn_add, 3, data_type);

        SimpleTensor<float> input1_dequantized = reference::dequantization_layer<float>(input1);
        SimpleTensor<float> input2_dequantized = reference::dequantization_layer<float>(input2);
        SimpleTensor<float> bn_mul_dequantized = reference::dequantization_layer<float>(bn_mul);
        SimpleTensor<float> bn_add_dequantized = reference::dequantization_layer<float>(bn_add);

        SimpleTensor<float> add_output_dequantized{ shape, DataType::F32 };
        SimpleTensor<float> bn_add_out_dequantized{ shape, DataType::F32 };

        reference::arithmetic_operation<float>(reference::ArithmeticOperation::ADD, input1_dequantized, input2_dequantized, add_output_dequantized, ConvertPolicy::SATURATE);
        SimpleTensor<float> bn_mul_out_dequantized = reference::pixel_wise_multiplication<float, float, float>(add_output_dequantized, bn_mul_dequantized, 1.f, ConvertPolicy::SATURATE,
                                                                                                               RoundingPolicy::TO_NEAREST_UP, DataType::F32);
        reference::arithmetic_operation<float>(reference::ArithmeticOperation::ADD, bn_mul_out_dequantized, bn_add_dequantized, bn_add_out_dequantized, ConvertPolicy::SATURATE);

        if(interm_out)
        {
            Parent::_interm_reference = reference::quantization_layer<float, T>(add_output_dequantized, data_type, Parent::_add_output_qinfo);
        }

        if(act_info.enabled() && act_info.activation() != ActivationLayerInfo::ActivationFunction::IDENTITY)
        {
            SimpleTensor<T> ref = reference::quantization_layer<float, T>(bn_add_out_dequantized, data_type, Parent::_final_output_qinfo);
            Parent::_reference  = reference::activation_layer(ref, act_info);
        }
        else
        {
            Parent::_reference = reference::quantization_layer<float, T>(bn_add_out_dequantized, data_type, Parent::_final_output_qinfo);
        }
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* TESTS_VALIDATION_FIXTURES_ADDMULADDFIXTURE */
