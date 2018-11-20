/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_NORMALIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_NORMALIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/NormalizationLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NormalizationValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, NormType norm_type, int norm_size, float beta, bool is_scaled, DataType data_type, DataLayout data_layout)
    {
        NormalizationLayerInfo info(norm_type, norm_size, 5, beta, 1.f, is_scaled);

        _target    = compute_target(shape, info, data_type, data_layout);
        _reference = compute_reference(shape, info, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(TensorShape shape, NormalizationLayerInfo info, DataType data_type, DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType norm_layer;
        norm_layer.configure(&src, &dst, info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        norm_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, NormalizationLayerInfo info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1 };

        // Fill reference
        fill(src);

        return reference::normalization_layer<T>(src, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NormalizationValidationFixture : public NormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, NormType norm_type, int norm_size, float beta, bool is_scaled, DataType data_type, DataLayout data_layout)
    {
        NormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, norm_type, norm_size, beta, is_scaled, data_type, data_layout);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_NORMALIZATION_LAYER_FIXTURE */
