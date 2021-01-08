/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/BatchNormalizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BatchNormalizationLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape0, TensorShape shape1, float epsilon, bool use_beta, bool use_gamma, ActivationLayerInfo act_info, DataType dt, DataLayout data_layout)
    {
        _data_type = dt;
        _use_beta  = use_beta;
        _use_gamma = use_gamma;

        _target    = compute_target(shape0, shape1, epsilon, act_info, dt, data_layout);
        _reference = compute_reference(shape0, shape1, epsilon, act_info, dt);
    }

protected:
    template <typename U>
    void fill(U &&src_tensor, U &&mean_tensor, U &&var_tensor, U &&beta_tensor, U &&gamma_tensor)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        const T          min_bound = T(-1.f);
        const T          max_bound = T(1.f);
        DistributionType distribution{ min_bound, max_bound };
        DistributionType distribution_var{ T(0.f), max_bound };

        library->fill(src_tensor, distribution, 0);
        library->fill(mean_tensor, distribution, 1);
        library->fill(var_tensor, distribution_var, 0);
        if(_use_beta)
        {
            library->fill(beta_tensor, distribution, 3);
        }
        else
        {
            // Fill with default value 0.f
            library->fill_tensor_value(beta_tensor, T(0.f));
        }
        if(_use_gamma)
        {
            library->fill(gamma_tensor, distribution, 4);
        }
        else
        {
            // Fill with default value 1.f
            library->fill_tensor_value(gamma_tensor, T(1.f));
        }
    }

    TensorType compute_target(TensorShape shape0, const TensorShape &shape1, float epsilon, ActivationLayerInfo act_info, DataType dt, DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape0, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src   = create_tensor<TensorType>(shape0, dt, 1, QuantizationInfo(), data_layout);
        TensorType dst   = create_tensor<TensorType>(shape0, dt, 1, QuantizationInfo(), data_layout);
        TensorType mean  = create_tensor<TensorType>(shape1, dt, 1);
        TensorType var   = create_tensor<TensorType>(shape1, dt, 1);
        TensorType beta  = create_tensor<TensorType>(shape1, dt, 1);
        TensorType gamma = create_tensor<TensorType>(shape1, dt, 1);

        // Create and configure function
        FunctionType norm;
        TensorType *beta_ptr  = _use_beta ? &beta : nullptr;
        TensorType *gamma_ptr = _use_gamma ? &gamma : nullptr;
        norm.configure(&src, &dst, &mean, &var, beta_ptr, gamma_ptr, epsilon, act_info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(mean.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(var.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(beta.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(gamma.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        mean.allocator()->allocate();
        var.allocator()->allocate();
        beta.allocator()->allocate();
        gamma.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!mean.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!var.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!beta.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!gamma.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), AccessorType(mean), AccessorType(var), AccessorType(beta), AccessorType(gamma));

        // Compute function
        norm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, float epsilon, ActivationLayerInfo act_info, DataType dt)
    {
        // Create reference
        SimpleTensor<T> ref_src{ shape0, dt, 1 };
        SimpleTensor<T> ref_mean{ shape1, dt, 1 };
        SimpleTensor<T> ref_var{ shape1, dt, 1 };
        SimpleTensor<T> ref_beta{ shape1, dt, 1 };
        SimpleTensor<T> ref_gamma{ shape1, dt, 1 };

        // Fill reference
        fill(ref_src, ref_mean, ref_var, ref_beta, ref_gamma);

        return reference::batch_normalization_layer(ref_src, ref_mean, ref_var, ref_beta, ref_gamma, epsilon, act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
    bool            _use_beta{};
    bool            _use_gamma{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FIXTURE */
