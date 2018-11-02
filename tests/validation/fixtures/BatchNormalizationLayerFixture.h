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
class BatchNormalizationLayerValidationFixedPointFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape0, TensorShape shape1, float epsilon, bool use_beta, bool use_gamma, ActivationLayerInfo act_info, DataType dt, DataLayout data_layout, int fractional_bits)
    {
        _fractional_bits = fractional_bits;
        _data_type       = dt;
        _use_beta        = use_beta;
        _use_gamma       = use_gamma;

        _target    = compute_target(shape0, shape1, epsilon, act_info, dt, data_layout, fractional_bits);
        _reference = compute_reference(shape0, shape1, epsilon, act_info, dt, fractional_bits);
    }

protected:
    template <typename U>
    void fill(U &&src_tensor, U &&mean_tensor, U &&var_tensor, U &&beta_tensor, U &&gamma_tensor)
    {
        if(is_data_type_float(_data_type))
        {
            float min_bound = 0.f;
            float max_bound = 0.f;
            std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<T>();
            std::uniform_real_distribution<> distribution(min_bound, max_bound);
            std::uniform_real_distribution<> distribution_var(0, max_bound);
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
                library->fill_tensor_value(beta_tensor, 0.f);
            }
            if(_use_gamma)
            {
                library->fill(gamma_tensor, distribution, 4);
            }
            else
            {
                // Fill with default value 1.f
                library->fill_tensor_value(gamma_tensor, 1.f);
            }
        }
        else
        {
            int min_bound = 0;
            int max_bound = 0;
            std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<T>(_fractional_bits);
            std::uniform_int_distribution<> distribution(min_bound, max_bound);
            std::uniform_int_distribution<> distribution_var(0, max_bound);
            library->fill(src_tensor, distribution, 0);
            library->fill(mean_tensor, distribution, 1);
            library->fill(var_tensor, distribution_var, 0);
            if(_use_beta)
            {
                library->fill(beta_tensor, distribution, 3);
            }
            else
            {
                // Fill with default value 0
                library->fill_tensor_value(beta_tensor, static_cast<T>(0));
            }
            if(_use_gamma)
            {
                library->fill(gamma_tensor, distribution, 4);
            }
            else
            {
                // Fill with default value 1
                library->fill_tensor_value(gamma_tensor, static_cast<T>(1 << (_fractional_bits)));
            }
        }
    }

    TensorType compute_target(TensorShape shape0, const TensorShape &shape1, float epsilon, ActivationLayerInfo act_info, DataType dt, DataLayout data_layout, int fixed_point_position)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape0, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src   = create_tensor<TensorType>(shape0, dt, 1, fixed_point_position, QuantizationInfo(), data_layout);
        TensorType dst   = create_tensor<TensorType>(shape0, dt, 1, fixed_point_position, QuantizationInfo(), data_layout);
        TensorType mean  = create_tensor<TensorType>(shape1, dt, 1, fixed_point_position);
        TensorType var   = create_tensor<TensorType>(shape1, dt, 1, fixed_point_position);
        TensorType beta  = create_tensor<TensorType>(shape1, dt, 1, fixed_point_position);
        TensorType gamma = create_tensor<TensorType>(shape1, dt, 1, fixed_point_position);

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

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, float epsilon, ActivationLayerInfo act_info, DataType dt, int fixed_point_position)
    {
        // Create reference
        SimpleTensor<T> ref_src{ shape0, dt, 1, fixed_point_position };
        SimpleTensor<T> ref_mean{ shape1, dt, 1, fixed_point_position };
        SimpleTensor<T> ref_var{ shape1, dt, 1, fixed_point_position };
        SimpleTensor<T> ref_beta{ shape1, dt, 1, fixed_point_position };
        SimpleTensor<T> ref_gamma{ shape1, dt, 1, fixed_point_position };

        // Fill reference
        fill(ref_src, ref_mean, ref_var, ref_beta, ref_gamma);

        return reference::batch_normalization_layer(ref_src, ref_mean, ref_var, ref_beta, ref_gamma, epsilon, act_info, fixed_point_position);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int             _fractional_bits{};
    DataType        _data_type{};
    bool            _use_beta{};
    bool            _use_gamma{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BatchNormalizationLayerValidationFixture : public BatchNormalizationLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape0, TensorShape shape1, float epsilon, bool use_beta, bool use_gamma, ActivationLayerInfo act_info, DataType dt, DataLayout data_layout)
    {
        BatchNormalizationLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(shape0, shape1, epsilon, use_beta, use_gamma, act_info, dt, data_layout, 0);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FIXTURE */
