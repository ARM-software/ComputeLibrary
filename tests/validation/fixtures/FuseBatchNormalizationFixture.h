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
#ifndef ARM_COMPUTE_TEST_FUSEBATCHNORMALIZATION_FIXTURE
#define ARM_COMPUTE_TEST_FUSEBATCHNORMALIZATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/FuseBatchNormalization.h"

#include <tuple>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, int dims_weights, typename T>
class FuseBatchNormalizationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_w, DataType data_type, DataLayout data_layout, bool in_place, bool with_bias, bool with_gamma, bool with_beta)
    {
        std::tie(_target_w, _target_b)       = compute_target(shape_w, data_type, data_layout, in_place, with_bias, with_gamma, with_beta);
        std::tie(_reference_w, _reference_b) = compute_reference(shape_w, data_type, data_layout, with_bias, with_gamma, with_beta);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        library->fill_tensor_uniform(tensor, i, min, max);
    }

    std::pair<TensorType, TensorType> compute_target(TensorShape shape_w, DataType data_type, DataLayout data_layout, bool in_place, bool with_bias, bool with_gamma, bool with_beta)
    {
        const TensorShape shape_v(shape_w[dims_weights - 1]);

        if(data_layout == DataLayout::NHWC)
        {
            permute(shape_w, PermutationVector(2U, 0U, 1U));
        }

        const bool in_place_w = in_place;
        const bool in_place_b = with_bias ? in_place : false;

        // Create tensors
        TensorType w       = create_tensor<TensorType>(shape_w, data_type, 1, QuantizationInfo(), data_layout);
        TensorType b       = create_tensor<TensorType>(shape_v, data_type);
        TensorType mean    = create_tensor<TensorType>(shape_v, data_type);
        TensorType var     = create_tensor<TensorType>(shape_v, data_type);
        TensorType w_fused = create_tensor<TensorType>(shape_w, data_type, 1, QuantizationInfo(), data_layout);
        TensorType b_fused = create_tensor<TensorType>(shape_v, data_type);
        TensorType beta    = create_tensor<TensorType>(shape_v, data_type);
        TensorType gamma   = create_tensor<TensorType>(shape_v, data_type);

        auto b_to_use       = with_bias ? &b : nullptr;
        auto gamma_to_use   = with_gamma ? &gamma : nullptr;
        auto beta_to_use    = with_beta ? &beta : nullptr;
        auto w_fused_to_use = in_place_w ? nullptr : &w_fused;
        auto b_fused_to_use = in_place_b ? nullptr : &b_fused;

        const FuseBatchNormalizationType fuse_bn_type = dims_weights == 3 ?
                                                        FuseBatchNormalizationType::DEPTHWISECONVOLUTION :
                                                        FuseBatchNormalizationType::CONVOLUTION;
        // Create and configure function
        FunctionType fuse_batch_normalization;
        fuse_batch_normalization.configure(&w, &mean, &var, w_fused_to_use, b_fused_to_use, b_to_use, beta_to_use, gamma_to_use, _epsilon, fuse_bn_type);

        ARM_COMPUTE_EXPECT(w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(mean.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(var.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(w_fused.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b_fused.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(beta.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(gamma.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        w.allocator()->allocate();
        b.allocator()->allocate();
        mean.allocator()->allocate();
        var.allocator()->allocate();
        w_fused.allocator()->allocate();
        b_fused.allocator()->allocate();
        beta.allocator()->allocate();
        gamma.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!mean.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!var.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!w_fused.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b_fused.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!beta.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!gamma.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(w), 0U, -1.0f, 1.0f);
        fill(AccessorType(b), 1U, -1.0f, 1.0f);
        fill(AccessorType(mean), 2U, -1.0f, 1.0f);
        fill(AccessorType(var), 3U, 0.0f, 1.0f);
        fill(AccessorType(beta), 4U, -1.0f, 1.0f);
        fill(AccessorType(gamma), 5U, -1.0f, 1.0f);

        // Compute function
        fuse_batch_normalization.run();

        return std::make_pair(std::move(in_place_w ? w : w_fused), std::move(in_place_b ? b : b_fused));
    }

    std::pair<SimpleTensor<T>, SimpleTensor<T>> compute_reference(TensorShape shape_w, DataType data_type, DataLayout data_layout, bool with_bias, bool with_gamma, bool with_beta)
    {
        const TensorShape shape_v(shape_w[dims_weights - 1]);

        SimpleTensor<T> w{ shape_w, data_type };
        SimpleTensor<T> b{ shape_v, data_type };
        SimpleTensor<T> mean{ shape_v, data_type };
        SimpleTensor<T> var{ shape_v, data_type };
        SimpleTensor<T> w_fused{ shape_w, data_type };
        SimpleTensor<T> b_fused{ shape_v, data_type };
        SimpleTensor<T> beta{ shape_v, data_type };
        SimpleTensor<T> gamma{ shape_v, data_type };

        // Fill reference tensor
        fill(w, 0U, -1.0f, 1.0f);
        fill(b, 1U, -1.0f, 1.0f);
        fill(mean, 2U, -1.0f, 1.0f);
        fill(var, 3U, 0.0f, 1.0f);
        fill(beta, 4U, -1.0f, 1.0f);
        fill(gamma, 5U, -1.0f, 1.0f);

        if(!with_bias)
        {
            // Fill with zeros
            fill(b, 0U, 0.0f, 0.0f);
        }

        if(!with_gamma)
        {
            // Fill with ones
            fill(gamma, 0U, 1.0f, 1.0f);
        }

        if(!with_beta)
        {
            // Fill with zeros
            fill(beta, 0U, 0.0f, 0.0f);
        }

        switch(dims_weights)
        {
            case 3:
                // Weights for depth wise convolution layer
                reference::fuse_batch_normalization_dwc_layer(w, mean, var, w_fused, b_fused, b, beta, gamma, _epsilon);
                break;
            case 4:
                // Weights for convolution layer
                reference::fuse_batch_normalization_conv_layer(w, mean, var, w_fused, b_fused, b, beta, gamma, _epsilon);
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported number of dimensions for the input weights tensor");
        }

        return std::make_pair(std::move(w_fused), std::move(b_fused));
    }

    const float     _epsilon{ 0.0001f };
    TensorType      _target_w{};
    TensorType      _target_b{};
    SimpleTensor<T> _reference_w{};
    SimpleTensor<T> _reference_b{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FUSEBATCHNORMALIZATION_FIXTURE */
