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
#ifndef ARM_COMPUTE_TEST_NORMALIZE_PLANAR_YUV_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_NORMALIZE_PLANAR_YUV_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/NormalizePlanarYUVLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NormalizePlanarYUVLayerValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape0, TensorShape shape1, DataType dt, DataLayout data_layout, QuantizationInfo quantization_info)
    {
        _data_type = dt;
        _target    = compute_target(shape0, shape1, dt, data_layout, quantization_info);
        _reference = compute_reference(shape0, shape1, dt, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&src_tensor, U &&mean_tensor, U &&std_tensor)
    {
        using FloatDistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<float>>::type;

        if(is_data_type_float(_data_type))
        {
            const T               min_bound = T(-1.f);
            const T               max_bound = T(1.f);
            FloatDistributionType distribution(min_bound, max_bound);
            FloatDistributionType distribution_std(T(0.1f), max_bound);
            library->fill(src_tensor, distribution, 0);
            library->fill(mean_tensor, distribution, 1);
            library->fill(std_tensor, distribution_std, 2);
        }
        else if(is_data_type_quantized_asymmetric(_data_type))
        {
            const QuantizationInfo quant_info = src_tensor.quantization_info();
            std::pair<int, int> bounds = get_quantized_bounds(quant_info, -1.f, 1.0f);
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);
            std::uniform_int_distribution<> distribution_std(quantize_qasymm8(0.1f, quant_info.uniform()), bounds.second);
            library->fill(src_tensor, distribution, 0);
            library->fill(mean_tensor, distribution, 1);
            library->fill(std_tensor, distribution_std, 2);
        }
    }

    TensorType compute_target(TensorShape shape0, const TensorShape &shape1, DataType dt, DataLayout data_layout, QuantizationInfo quantization_info)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape0, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src  = create_tensor<TensorType>(shape0, dt, 1, quantization_info, data_layout);
        TensorType mean = create_tensor<TensorType>(shape1, dt, 1, quantization_info);
        TensorType std  = create_tensor<TensorType>(shape1, dt, 1, quantization_info);
        TensorType dst;

        // Create and configure function
        FunctionType norm;
        norm.configure(&src, &dst, &mean, &std);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(mean.info()->is_resizable());
        ARM_COMPUTE_ASSERT(std.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        mean.allocator()->allocate();
        std.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!mean.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!std.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), AccessorType(mean), AccessorType(std));

        // Compute function
        norm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, DataType dt, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> ref_src{ shape0, dt, 1, quantization_info };
        SimpleTensor<T> ref_mean{ shape1, dt, 1, quantization_info };
        SimpleTensor<T> ref_std{ shape1, dt, 1, quantization_info };

        // Fill reference
        fill(ref_src, ref_mean, ref_std);

        return reference::normalize_planar_yuv_layer(ref_src, ref_mean, ref_std);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NormalizePlanarYUVLayerValidationFixture : public NormalizePlanarYUVLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape0, TensorShape shape1, DataType dt, DataLayout data_layout)
    {
        NormalizePlanarYUVLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape0, shape1, dt, data_layout, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NormalizePlanarYUVLayerValidationQuantizedFixture : public NormalizePlanarYUVLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape0, TensorShape shape1, DataType dt, DataLayout data_layout, QuantizationInfo quantization_info)
    {
        NormalizePlanarYUVLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape0, shape1, dt, data_layout, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_NORMALIZE_PLANAR_YUV_LAYER_FIXTURE */
