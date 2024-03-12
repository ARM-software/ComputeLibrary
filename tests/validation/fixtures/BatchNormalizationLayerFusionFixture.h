/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FUSION_FIXTURE
#define ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FUSION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/BatchNormalizationLayer.h"
#include "tests/validation/reference/ConvolutionLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename ConvolutionFunctionType, typename FusionFunctionType, typename T>
class BatchNormalizationLayerFusionValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape src_shape, TensorShape w_shape, TensorShape b_shape, TensorShape dst_shape, PadStrideInfo info, Size2D dilation,
               bool use_conv_b, bool use_beta, bool use_gamma, float epsilon, DataType dt, DataLayout data_layout)
    {
        ARM_COMPUTE_UNUSED(dilation);

        _data_type   = dt;
        _data_layout = data_layout;
        _use_conv_b  = use_conv_b;
        _use_beta    = use_beta;
        _use_gamma   = use_gamma;

        _target    = compute_target(src_shape, w_shape, b_shape, dst_shape, info, epsilon);
        _reference = compute_reference(src_shape, w_shape, b_shape, dst_shape, info, epsilon);
    }

protected:
    template <typename U>
    void fill(U &&src, U &&w_tensor, U &&b_tensor, U &&mean_tensor, U &&var_tensor, U &&beta_tensor, U &&gamma_tensor)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.f), T(1.f) };
        DistributionType distribution_gz{ T(0.f), T(1.f) };

        library->fill(src, distribution, 0);
        library->fill(w_tensor, distribution, 1);
        library->fill(mean_tensor, distribution, 2);
        library->fill(var_tensor, distribution_gz, 3);
        _use_conv_b ? library->fill(b_tensor, distribution, 4) : library->fill_tensor_value(b_tensor, T(0.f));
        _use_beta ? library->fill(beta_tensor, distribution, 5) : library->fill_tensor_value(beta_tensor, T(0.f));
        _use_gamma ? library->fill(gamma_tensor, distribution, 6) : library->fill_tensor_value(gamma_tensor, T(1.f));
    }

    TensorType compute_target(TensorShape src_shape, TensorShape w_shape, TensorShape b_shape, TensorShape dst_shape, PadStrideInfo info, float epsilon)
    {
        if(_data_layout == DataLayout::NHWC)
        {
            permute(src_shape, PermutationVector(2U, 0U, 1U));
            permute(w_shape, PermutationVector(2U, 0U, 1U));
            permute(dst_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src      = create_tensor<TensorType>(src_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType conv_w   = create_tensor<TensorType>(w_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType conv_b   = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bn_mean  = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bn_var   = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bn_beta  = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bn_gamma = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType fused_w  = create_tensor<TensorType>(w_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType fused_b  = create_tensor<TensorType>(b_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType dst      = create_tensor<TensorType>(dst_shape, _data_type, 1, QuantizationInfo(), _data_layout);

        // Create and configure function
        FusionFunctionType      fuse_fn;
        ConvolutionFunctionType conv_fn;
        TensorType             *conv_b_ptr = _use_conv_b ? &conv_b : nullptr;
        TensorType             *beta_ptr   = _use_beta ? &bn_beta : nullptr;
        TensorType             *gamma_ptr  = _use_gamma ? &bn_gamma : nullptr;
        fuse_fn.configure(&conv_w, &bn_mean, &bn_var, &fused_w, &fused_b, conv_b_ptr, beta_ptr, gamma_ptr, epsilon);
        conv_fn.configure(&src, &fused_w, &fused_b, &dst, info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(conv_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(conv_b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bn_mean.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bn_var.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bn_beta.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bn_gamma.info()->is_resizable());
        ARM_COMPUTE_ASSERT(fused_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(fused_b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        conv_w.allocator()->allocate();
        conv_b.allocator()->allocate();
        bn_mean.allocator()->allocate();
        bn_var.allocator()->allocate();
        bn_beta.allocator()->allocate();
        bn_gamma.allocator()->allocate();
        fused_w.allocator()->allocate();
        fused_b.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!conv_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!conv_b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bn_mean.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bn_var.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bn_beta.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bn_gamma.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!fused_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!fused_b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src),
             AccessorType(conv_w), AccessorType(conv_b),
             AccessorType(bn_mean), AccessorType(bn_var), AccessorType(bn_beta), AccessorType(bn_gamma));

        // Compute function
        fuse_fn.run();
        conv_fn.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(TensorShape src_shape, TensorShape w_shape, TensorShape b_shape, TensorShape dst_shape, PadStrideInfo info, float epsilon)
    {
        // Create reference
        SimpleTensor<T> src{ src_shape, _data_type, 1 };
        SimpleTensor<T> conv_w{ w_shape, _data_type, 1 };
        SimpleTensor<T> conv_b{ b_shape, _data_type, 1 };
        SimpleTensor<T> bn_var{ b_shape, _data_type, 1 };
        SimpleTensor<T> bn_mean{ b_shape, _data_type, 1 };
        SimpleTensor<T> bn_beta{ b_shape, _data_type, 1 };
        SimpleTensor<T> bn_gamma{ b_shape, _data_type, 1 };

        // Fill reference
        fill(src, conv_w, conv_b, bn_mean, bn_var, bn_beta, bn_gamma);

        // Calculate Conv + BN
        auto conv_res = reference::convolution_layer(src, conv_w, conv_b, dst_shape, info);
        return reference::batch_normalization_layer(conv_res, bn_mean, bn_var, bn_beta, bn_gamma, epsilon, ActivationLayerInfo());
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
    DataLayout      _data_layout{};
    bool            _use_conv_b{};
    bool            _use_beta{};
    bool            _use_gamma{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_NORMALIZATION_LAYER_FUSION_FIXTURE */
