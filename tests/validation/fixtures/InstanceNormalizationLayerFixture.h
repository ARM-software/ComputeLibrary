/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_INSTANCENORMALIZATION_FIXTURE
#define ARM_COMPUTE_TEST_INSTANCENORMALIZATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/InstanceNormalizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class InstanceNormalizationLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataLayout data_layout, bool in_place)
    {
        _target    = compute_target(shape, data_type, data_layout, in_place);
        _reference = compute_reference(shape, data_type, data_layout);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        std::uniform_real_distribution<> distribution(1.f, 2.f);
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(TensorShape shape, DataType data_type, DataLayout data_layout, bool in_place)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> dist_gamma(1.f, 2.f);
        std::uniform_real_distribution<float> dist_beta(-2.f, 2.f);
        std::uniform_real_distribution<float> dist_epsilon(1e-16f, 1e-12f);

        const float gamma   = dist_gamma(gen);
        const float beta    = dist_beta(gen);
        const float epsilon = dist_epsilon(gen);

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType instance_norm_func;
        instance_norm_func.configure(&src, in_place ? nullptr : &dst, gamma, beta, epsilon);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        if(!in_place)
        {
            ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Allocate tensors
        src.allocator()->allocate();
        if(!in_place)
        {
            dst.allocator()->allocate();
        }

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        if(!in_place)
        {
            ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        instance_norm_func.run();

        if(in_place)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, DataLayout data_layout)
    {
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> dist_gamma(1.f, 2.f);
        std::uniform_real_distribution<float> dist_beta(-2.f, 2.f);
        std::uniform_real_distribution<float> dist_epsilon(1e-16f, 1e-12f);

        const float gamma   = dist_gamma(gen);
        const float beta    = dist_beta(gen);
        const float epsilon = dist_epsilon(gen);

        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        return reference::instance_normalization<T>(src, gamma, beta, epsilon);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTANCENORMALIZATION_FIXTURE */
