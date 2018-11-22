/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_YOLO_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_YOLO_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/YOLOLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class YOLOValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, int32_t num_classes, DataLayout data_layout, DataType data_type,
               QuantizationInfo quantization_info)
    {
        _data_type = data_type;
        _function  = function;

        ActivationLayerInfo info(function, alpha_beta, alpha_beta);

        _target    = compute_target(shape, in_place, info, num_classes, data_layout, data_type, quantization_info);
        _reference = compute_reference(shape, info, num_classes, data_type, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        float min_bound = 0;
        float max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<T>(_function, _data_type);
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(TensorShape shape, bool in_place, const ActivationLayerInfo &info, int32_t num_classes, DataLayout data_layout, DataType data_type, QuantizationInfo quantization_info)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, quantization_info, data_layout);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, quantization_info, data_layout);

        // Create and configure function
        FunctionType yolo_layer;

        TensorType *dst_ptr = in_place ? &src : &dst;

        yolo_layer.configure(&src, dst_ptr, info, num_classes);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        if(!in_place)
        {
            dst.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        yolo_layer.run();

        if(in_place)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const ActivationLayerInfo &info, int32_t num_classes, DataType data_type, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, quantization_info };

        // Fill reference
        fill(src);

        return reference::yolo_layer<T>(src, info, num_classes);
    }

    TensorType                              _target{};
    SimpleTensor<T>                         _reference{};
    DataType                                _data_type{};
    ActivationLayerInfo::ActivationFunction _function{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class YOLOValidationFixture : public YOLOValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, int32_t num_classes, DataLayout data_layout, DataType data_type)
    {
        YOLOValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, in_place, function, alpha_beta, num_classes, data_layout, data_type, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class YOLOValidationQuantizedFixture : public YOLOValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, bool in_place, ActivationLayerInfo::ActivationFunction function, float alpha_beta, int32_t num_classes, DataLayout data_layout, DataType data_type,
               QuantizationInfo quantization_info)
    {
        YOLOValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, in_place, function, alpha_beta, num_classes, data_layout, data_type, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ARM_COMPUTE_TEST_YOLO_LAYER_FIXTURE
