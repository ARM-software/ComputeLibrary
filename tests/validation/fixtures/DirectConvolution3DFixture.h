/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/Conv3D.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolution3DValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &input_shape, int stride_x, int stride_y, int stride_z, int pad_x, int pad_y, int pad_z, unsigned int kernel_width, int kernel_height, int kernel_depth,
               unsigned int num_kernels, bool has_bias, const ActivationLayerInfo &act_info, const DataType &data_type, const DataLayout &data_layout)
    {
        ARM_COMPUTE_ERROR_ON(data_layout != DataLayout::NDHWC);

        const TensorShape weights_shape(num_kernels, input_shape[0], kernel_width, kernel_height, kernel_depth);
        const TensorShape bias_shape(num_kernels);
        const Conv3dInfo  conv3d_info(Size3D(stride_x, stride_y, stride_z), Padding3D(pad_x, pad_y, pad_z), act_info, Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false);
        const TensorShape output_shape = compute_conv3d_shape(input_shape, weights_shape, conv3d_info);

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, conv3d_info, has_bias, data_type, data_layout);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, conv3d_info, has_bias, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const Conv3dInfo &conv3d_info,
                              bool has_bias, const DataType &data_type, const DataLayout &data_layout)
    {
        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType bias    = has_bias ? create_tensor<TensorType>(bias_shape, data_type, 1, QuantizationInfo()) : TensorType();
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType conv{};
        conv.configure(&src, &weights, has_bias ? &bias : nullptr, &dst, conv3d_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);

        if(has_bias)
        {
            ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
            bias.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
            fill(AccessorType(bias), 2);
        }

        // Compute Direct Convolution 3D function
        conv.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const Conv3dInfo &conv3d_info,
                                      bool has_bias, const DataType &data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type };
        SimpleTensor<T> weights{ weights_shape, data_type };
        SimpleTensor<T> bias{ bias_shape, data_type };
        SimpleTensor<T> dst{ output_shape, data_type };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);

        if(has_bias)
        {
            fill(bias, 2);
        }

        return reference::activation_layer(reference::conv3d<T>(src, weights, bias, dst, conv3d_info), conv3d_info.act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolution3DValidationFixture : public DirectConvolution3DValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int stride_z, int pad_x, int pad_y, int pad_z, unsigned int kernel_width, int kernel_height, int kernel_depth,
               unsigned int num_kernels, bool has_bias, ActivationLayerInfo act_info, DataType data_type, DataLayout data_layout)
    {
        DirectConvolution3DValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, stride_x, stride_y, stride_z, pad_x, pad_y, pad_z, kernel_width, kernel_height,
                                                                                                      kernel_depth, num_kernels, has_bias, act_info, data_type, data_layout);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
