/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_MATMULFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_MATMULFIXTURE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "src/core/utils/quantization/AsymmHelpers.h"
#include "tests/framework/Asserts.h" // Required for ARM_COMPUTE_ASSERT
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/GEMMLowp.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/ReshapeLayer.h"
#include "tests/validation/Validation.h"

#include <limits>
#include <random>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulGenericValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         output_shape,
               bool                transpose_a,
               bool                transpose_b,
               DataType            data_type,
               ActivationLayerInfo act_info,
               int                 num_extra_runs,
               Settings            settings,
               QuantizationInfo    a_qinfo = QuantizationInfo(),
               QuantizationInfo    b_qinfo = QuantizationInfo(),
               QuantizationInfo    o_qinfo = QuantizationInfo())
    {
        if(std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        // For brevity, the input shapes are assumed to be not-transposed for both a and b matrices.
        if (transpose_a)
        {
            permute(shape_a, PermutationVector(1U, 0U));
        }
        if (transpose_b)
        {
            permute(shape_b, PermutationVector(1U, 0U));
        }

        _target    = compute_target(shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info,
                                    num_extra_runs, settings, a_qinfo, b_qinfo, o_qinfo);
        _reference = compute_reference(shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info,
                                       a_qinfo, b_qinfo, o_qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float lo = -1.f, float hi = 1.f)
    {
        switch (tensor.data_type())
        {
            case DataType::BFLOAT16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<bfloat16> distribution{float(lo), float(hi)};
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{float(lo), float(hi)};
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(lo, hi);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
            {
                library->fill_tensor_uniform(tensor, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported data type.");
            }
        }
    }

    virtual TensorType compute_target(const TensorShape  &shape_a,
                                      const TensorShape  &shape_b,
                                      const TensorShape  &output_shape,
                                      bool                transpose_a,
                                      bool                transpose_b,
                                      DataType            data_type,
                                      ActivationLayerInfo act_info,
                                      int                 num_extra_runs,
                                      const Settings     &settings,
                                      QuantizationInfo    a_qinfo,
                                      QuantizationInfo    b_qinfo,
                                      QuantizationInfo    o_qinfo)
    {
        // 1. Create Classes and configure function
        // ----------------------------------------------------
        // Create tensors
        // Configure relevant classes and matmul function
        TensorType a   = create_tensor<TensorType>(shape_a, data_type, 1, a_qinfo);
        TensorType b   = create_tensor<TensorType>(shape_b, data_type, 1, b_qinfo);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, o_qinfo);

        FunctionType matmul;

        // Configure MatMulInfo class
        MatMulInfo mm_info;
        mm_info.adj_lhs(transpose_a).adj_rhs(transpose_b);

        // Ensure values are dynamic
        a.info()->set_are_values_constant(false);
        b.info()->set_are_values_constant(false);

        // Configure operator
        matmul.configure(&a, &b, &dst, mm_info, settings, act_info);

        // Assertions
        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // For multiple runs.
        for (int i = 0; i < num_extra_runs; i++)
        {
            // Stress dynamic tensors by running multiple times.
            // --------------------------------------------------------
            // Fill tensors with new seed
            // Run function
            const int seed_offset = num_extra_runs * 100;
            fill(AccessorType(a), seed_offset);
            fill(AccessorType(b), seed_offset + 1);

            matmul.run();
        }

        // 2. Final Run for reference comparison
        // --------------------------------------------------------
        // Re-fill tensors same seed as reference run
        // Compute MatMul operation
        fill(AccessorType(a), 2);
        fill(AccessorType(b), 3);

        matmul.run();

        return dst;
    }

    template <typename TT>
    typename std::enable_if < !std::is_integral<TT>::value, SimpleTensor<TT >>::type
                                                                            compute_reference_gemm(const SimpleTensor<TT> &a,
                                                                                                   const SimpleTensor<TT> &b,
                                                                                                   const SimpleTensor<TT> &c,
                                                                                                   float                   alpha,
                                                                                                   float                   beta,
                                                                                                   const QuantizationInfo &o_qinfo)
    {
        ARM_COMPUTE_UNUSED(o_qinfo);

        return reference::gemm(a, b, c, alpha, beta);
    }

    template <typename TT>
    typename std::enable_if<std::is_integral<TT>::value, SimpleTensor<TT>>::type
                                                                        compute_reference_gemm(const SimpleTensor<TT> &a,
                                                                                               const SimpleTensor<TT> &b,
                                                                                               const SimpleTensor<TT> &c,
                                                                                               float                   alpha,
                                                                                               float                   beta,
                                                                                               const QuantizationInfo &o_qinfo)
    {
        ARM_COMPUTE_UNUSED(alpha, beta);

        const auto aq = a.quantization_info().uniform();
        const auto bq = b.quantization_info().uniform();
        const auto oq = o_qinfo.uniform();

        const auto multiplier = aq.scale * bq.scale / oq.scale;

        int32_t output_multiplier = 0;
        int32_t output_shift      = 0;
        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
        std::vector<int32_t> output_multipliers{output_multiplier};
        std::vector<int32_t> output_shifts{output_shift};

        //The lhs and rhs offsets are negated here to keep the reference aligned with the function implementation where the lhs and rhs offsets are also negated.
        const auto tmp = reference::gemmlowp_matrix_multiply_core<int32_t>(a, b, c.shape(), -aq.offset, -bq.offset);

        auto output = reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, TT>(
                          tmp, output_multipliers, output_shifts, oq.offset, std::numeric_limits<int32_t>::lowest(),
                          std::numeric_limits<int32_t>::max());
        output.quantization_info(o_qinfo);

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape  &a_shape,
                                      const TensorShape  &b_shape,
                                      const TensorShape  &output_shape,
                                      bool                transpose_a,
                                      bool                transpose_b,
                                      DataType            data_type,
                                      ActivationLayerInfo act_info,
                                      QuantizationInfo    a_qinfo,
                                      QuantizationInfo    b_qinfo,
                                      QuantizationInfo    o_qinfo)
    {
        // We collapse dimensions > 2 onto dimension 2, i.e. 4D+ tensors will look like 3D
        // This is necessary unless we choose to extend gemm reference for 4D+ tensors
        TensorShape output_shape_collapsed = output_shape.collapsed_from(Window::DimZ);
        TensorShape a_shape_collapsed      = a_shape.collapsed_from(Window::DimZ);
        TensorShape b_shape_collapsed      = b_shape.collapsed_from(Window::DimZ);

        // Create reference
        SimpleTensor<T> a{a_shape_collapsed, data_type, 1, a_qinfo};
        SimpleTensor<T> b{b_shape_collapsed, data_type, 1, b_qinfo};
        SimpleTensor<T> c{output_shape_collapsed, data_type, 1};

        // Fill reference
        fill(a, 2);
        fill(b, 3);

        /* Note: Assuming the usual batch matmul dimensions A = (B x M x K), B = (B x K x N), if transpose_a is set to true, then A is assumed to be (B x K x M),
        therefore, A must be pre-transposed before passing it to the fixture. And, we transpose A again in the fixture to make it (B x M x K)
        in order to be able to call reference implementation that works with (B x M x K) input.
        Similarly, if transpose_b is set to true, then B is assumed to be (B x N x K), B must be pre-transposed before passing it to the fixture. */

        // Define transposed shapes
        TensorShape a_transposed_shape(a.shape());
        a_transposed_shape.set(0, a.shape().y());
        a_transposed_shape.set(1, a.shape().x());

        TensorShape b_transposed_shape(b.shape());
        b_transposed_shape.set(0, b.shape().y());
        b_transposed_shape.set(1, b.shape().x());

        // Define transposed tensors
        SimpleTensor<T> a_transposed{a_transposed_shape, data_type};
        SimpleTensor<T> b_transposed{b_transposed_shape, data_type};

        // pretranspose a if necessary
        if (transpose_a)
        {
            a_transposed = reference::permute<T>(a, PermutationVector(1U, 0U));
        }
        // pretranspose b if necessary
        if (transpose_b)
        {
            b_transposed = reference::permute<T>(b, PermutationVector(1U, 0U));
        }

        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: alpha * A * B + 0 * C
        // Use transposed tensors if boolean enabled else use original tensors
        auto result = compute_reference_gemm<T>((transpose_a) ? a_transposed : a, (transpose_b) ? b_transposed : b, c,
                                                1.0f, 0.f, o_qinfo);

        result = reference::activation_layer<T>(result, act_info, o_qinfo);

        // We reshape the gemm output back if the tensor is high dimensional
        if (output_shape_collapsed != output_shape)
        {
            result = reference::reshape_layer(result, output_shape);
        }

        return result;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

/// TODO: (ONCPUML-1451) The current state of this fixture is interim and a longer-term testing method will be implemented later.
/// @note: Currently we support only a 2x2 test due to the lack of reorder ref. implementation.
template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulFixedFormatFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    TensorType compute_target(const TensorShape  &shape_a,
                              const TensorShape  &shape_b,
                              const TensorShape  &output_shape,
                              bool                transpose_a,
                              bool                transpose_b,
                              DataType            data_type,
                              ActivationLayerInfo act_info,
                              int                 num_extra_runs,
                              const Settings     &settings,
                              QuantizationInfo    a_qinfo,
                              QuantizationInfo    b_qinfo,
                              QuantizationInfo    o_qinfo) override
    {
        // 1. Create Classes and configure function
        // ----------------------------------------------------
        // Create tensors
        // Configure relevant classes and matmul function
        TensorType a   = create_tensor<TensorType>(shape_a, data_type, 1, a_qinfo);
        TensorType b   = create_tensor<TensorType>(shape_b, data_type, 1, b_qinfo);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, o_qinfo);

        const auto       weight_tensor_info  = TensorInfo(*b.info());
        const TensorInfo new_tensor_info     = prepare_weights(weight_tensor_info);
        TensorType       weights_transformed = create_tensor<TensorType>(new_tensor_info);

        // Configure MatMulInfo class
        MatMulInfo mm_info;
        mm_info.adj_lhs(transpose_a).adj_rhs(transpose_b);

        // Ensure values are dynamic
        a.info()->set_are_values_constant(false);
        b.info()->set_are_values_constant(false);
        weights_transformed.info()->set_are_values_constant(false);

        FunctionType matmul;

        // Configure operator
        matmul.configure(&a, &weights_transformed, &dst, mm_info, settings, act_info);

        // Assertions
        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights_transformed.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        dst.allocator()->allocate();
        weights_transformed.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights_transformed.info()->is_resizable());

        // For multiple runs.
        for (int i = 0; i < num_extra_runs; i++)
        {
            // Stress dynamic tensors by running multiple times.
            // --------------------------------------------------------
            // Fill tensors with new seed
            // Run function
            const int seed_offset = num_extra_runs * 100;
            this->fill(AccessorType(a), seed_offset);
            this->fill(AccessorType(b), seed_offset + 1);

            matmul.run();
        }

        // 2. Final Run for reference comparison
        // --------------------------------------------------------
        // Re-fill tensors same seed as reference run
        // Compute MatMul operation
        this->fill(AccessorType(a), 2);
        this->fill(AccessorType(b), 3);

        rearrange_data(AccessorType(b), AccessorType(weights_transformed));

        matmul.run();

        return dst;
    }

    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         output_shape,
               bool                transpose_a,
               bool                transpose_b,
               DataType            data_type,
               ActivationLayerInfo act_info,
               int                 num_extra_runs,
               Settings            settings,
               QuantizationInfo    a_qinfo,
               QuantizationInfo    b_qinfo,
               QuantizationInfo    o_qinfo)
    {
        if (CPUInfo::get().has_bf16())
        {
            MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
                shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, num_extra_runs, settings,
                a_qinfo, b_qinfo, o_qinfo);
        }
    }

private:
    TensorInfo prepare_weights(const TensorInfo tensor_info)
    {
        const DataLayout data_layout = tensor_info.data_layout();
        ARM_COMPUTE_EXPECT(data_layout == DataLayout::NCHW, framework::LogLevel::ERRORS);
        const DataType    data_type    = tensor_info.data_type();
        const TensorShape tensor_shape = tensor_info.tensor_shape();
        const int         H = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)];
        const int         W = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)];
        ARM_COMPUTE_EXPECT(H <= 2 && W <= 2, framework::LogLevel::ERRORS);

        arm_compute::Strides strides_in_bytes = tensor_info.strides_in_bytes();
        strides_in_bytes.set(1, 32);
        strides_in_bytes.set(2, 32);

        const size_t offset_first_element_in_bytes = tensor_info.offset_first_element_in_bytes();
        const size_t total_size_in_bytes           = 32;

        const TensorShape TS(H, W);

        TensorInfo new_tensor_info = tensor_info;
        new_tensor_info.init(TS, tensor_info.num_channels(), data_type, strides_in_bytes, offset_first_element_in_bytes,
                             total_size_in_bytes);

        return new_tensor_info;
    }

    void rearrange_data(const AccessorType src, AccessorType dst)
    {
        const TensorShape src_tensor_shape = src.shape();
        const DataLayout  data_layout      = src.data_layout();
        ARM_COMPUTE_EXPECT(data_layout == DataLayout::NCHW, framework::LogLevel::ERRORS);
        const unsigned int O =
            src_tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES)]; // N=O
        const unsigned int H =
            src_tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)];
        const unsigned int W =
            src_tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)];
        const unsigned int I =
            src_tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL)]; // C=I
        ARM_COMPUTE_EXPECT(H <= 2 && W <= 2, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(I == 1 && O == 1, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(src.num_elements() <= dst.num_elements(), framework::LogLevel::ERRORS);

        const T *src_ptr = reinterpret_cast<const T *>(src.data());
        T       *dst_ptr = reinterpret_cast<T *>(dst.data());

        // rearrange indexes for 2x2 input and weight
        int dst_idx[] = {0, 4, 1, 5};
        for (int i = 0; i < 4; i++)
        {
            dst_ptr[dst_idx[i]] = src_ptr[i];
        }
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulValidationFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape output_shape,
               bool        transpose_a,
               bool        transpose_b,
               DataType    data_type)
    {
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, ActivationLayerInfo(), 0, Settings());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulValidationWithDynamicTensorsFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         output_shape,
               bool                transpose_a,
               bool                transpose_b,
               DataType            data_type,
               ActivationLayerInfo act_info,
               int                 num_extra_runs)
    {
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, num_extra_runs, Settings());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class QuantizedMatMulValidationFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         output_shape,
               bool                transpose_a,
               bool                transpose_b,
               DataType            data_type,
               ActivationLayerInfo act_info,
               int                 num_extra_runs,
               QuantizationInfo    a_qinfo,
               QuantizationInfo    b_qinfo,
               QuantizationInfo    o_qinfo)
    {
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, num_extra_runs, Settings(),
            a_qinfo, b_qinfo, o_qinfo);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulValidationWithActivationFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         output_shape,
               bool                transpose_a,
               bool                transpose_b,
               DataType            data_type,
               ActivationLayerInfo act_info)
    {
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, 0, Settings());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class MatMulValidationWithActivationAlphaBetaFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape                             shape_a,
               TensorShape                             shape_b,
               TensorShape                             output_shape,
               bool                                    transpose_a,
               bool                                    transpose_b,
               DataType                                data_type,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta)
    {
        ActivationLayerInfo act_info(function, alpha_beta, alpha_beta);
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, 0, Settings());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename Settings, typename T>
class QuantizedMatMulValidationWithActivationFixture
    : public MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>
{
public:
    void setup(TensorShape                             shape_a,
               TensorShape                             shape_b,
               TensorShape                             output_shape,
               bool                                    transpose_a,
               bool                                    transpose_b,
               DataType                                data_type,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               int                                     num_extra_runs,
               QuantizationInfo                        a_qinfo,
               QuantizationInfo                        b_qinfo,
               QuantizationInfo                        o_qinfo)
    {
        ActivationLayerInfo act_info(function, alpha_beta, alpha_beta);
        MatMulGenericValidationFixture<TensorType, AccessorType, FunctionType, Settings, T>::setup(
            shape_a, shape_b, output_shape, transpose_a, transpose_b, data_type, act_info, num_extra_runs, Settings(),
            a_qinfo, b_qinfo, o_qinfo);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_MATMULFIXTURE_H
