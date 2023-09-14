/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_MATMULKERNELFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_MATMULKERNELFIXTURE_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/GEMMLowp.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/ReshapeLayer.h"
#include <cmath>
#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::opencl::kernels;

template <typename T, typename KernelType, bool use_mmul = false>
class MatMulKernelGenericValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape output_shape, bool pretranspose_a, bool pretranspose_b, int M0, int N0, int K0, bool export_rhs_to_cl_image, DataType data_type,
               bool enable_bias)
    {
        // This hash is used by random generators. There may be hash collisions but
        // this is intentional as it's a very easy way to make the the current
        // random generation process almost different for many test configurations,
        // which were using the same set of values before.
        _hash = M0 + N0 + K0 + shape_a[0] + shape_a[1] + shape_b[0] + shape_b[1] + enable_bias + export_rhs_to_cl_image;

        // Flag to create a bias
        _enable_bias = enable_bias;

        // For brevity, the input shapes are assumed to be not-transposed for both Lhs and Rhs matrices.
        QuantizationInfo lhs_q_info;
        QuantizationInfo rhs_q_info;
        QuantizationInfo dst_q_info;

        if(is_data_type_quantized(data_type))
        {
            const int32_t t_max = static_cast<int32_t>(std::numeric_limits<T>::max());
            const int32_t t_min = static_cast<int32_t>(std::numeric_limits<T>::min());

            std::mt19937                           generator(library->seed() + _hash);
            std::uniform_real_distribution<float>  distribution_float(-5.0f, 3.0f);
            std::uniform_int_distribution<int32_t> distribution_t(t_min, t_max);

            const float scale_lhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
            const float scale_rhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]

            const int32_t offset_lhs = distribution_t(generator);
            const int32_t offset_rhs = distribution_t(generator);

            lhs_q_info = QuantizationInfo(scale_lhs, offset_lhs);
            rhs_q_info = QuantizationInfo(scale_rhs, offset_rhs);

            const int m = shape_a.y();
            const int n = shape_b.x();
            const int k = shape_a.x();

            const float bias_fraction = enable_bias ? 0.5f : 0.f;

            QuantizationHint q_hint = suggest_matmul_dst_q_info_and_bias(lhs_q_info, rhs_q_info, m, n, k, data_type, bias_fraction);
            dst_q_info              = q_hint.q_info;
            _min_bias               = q_hint.bias_min;
            _max_bias               = q_hint.bias_max;
        }

        if(pretranspose_a)
        {
            permute(shape_a, PermutationVector(1U, 0U));
        }

        if(pretranspose_b)
        {
            permute(shape_b, PermutationVector(1U, 0U));
        }

        // Skip configurations unsupported by the device.
        _device_supports_export_to_cl_image = image2d_from_buffer_supported(CLKernelLibrary::get().get_device());
        if(!_device_supports_export_to_cl_image && export_rhs_to_cl_image)
        {
            ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
            framework::ARM_COMPUTE_PRINT_INFO();
            return; // Note: Also need to skip the validate in corresponding FIXTURE_DATA_TEST_CASEs.
        }

        _device_supports_mmul = arm_matrix_multiply_supported(CLKernelLibrary::get().get_device());
        if(!_device_supports_mmul && use_mmul)
        {
            ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
            framework::ARM_COMPUTE_PRINT_INFO();
            return; // Note: Also need to skip the validate in corresponding FIXTURE_DATA_TEST_CASEs.
        }

        _target    = compute_target(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, M0, N0, K0, export_rhs_to_cl_image, data_type, lhs_q_info, rhs_q_info, dst_q_info);
        _reference = compute_reference(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, data_type, lhs_q_info, rhs_q_info, dst_q_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float lo = -1.f, float hi = 1.f)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ float(lo), float(hi) };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(lo, hi);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    template <typename U>
    void fill_bias_s32(U &&tensor, int i, int32_t min, int32_t max)
    {
        std::uniform_int_distribution<int32_t> distribution(min, max);
        library->fill(tensor, distribution, i);
    }

    template <typename U, typename D>
    void fill_constant(U &&tensor, D value)
    {
        library->fill_tensor_value(tensor, value);
    }

    CLTensor compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &output_shape, bool pretranspose_a, bool pretranspose_b, const int M0, const int N0, const int K0,
                            bool export_rhs_to_cl_image, DataType data_type, const QuantizationInfo &lhs_q_info, const QuantizationInfo &rhs_q_info, const QuantizationInfo &dst_q_info)
    {
        CLSynthetizeOperator<KernelType> matMul{};
        MatMulKernelInfo                 matmul_info;
        matmul_info.adj_lhs                = pretranspose_a;
        matmul_info.adj_rhs                = pretranspose_b;
        matmul_info.m0                     = M0;
        matmul_info.n0                     = N0;
        matmul_info.k0                     = K0;
        matmul_info.export_rhs_to_cl_image = export_rhs_to_cl_image;

        bool is_quantized = is_data_type_quantized(data_type);

        // Create tensors
        CLTensor a    = create_tensor<CLTensor>(shape_a, data_type, 1, lhs_q_info);
        CLTensor b    = create_tensor<CLTensor>(shape_b, data_type, 1, rhs_q_info);
        CLTensor bias = create_tensor<CLTensor>(output_shape[0], (is_quantized) ? DataType::S32 : data_type, 1, dst_q_info);
        CLTensor dst  = create_tensor<CLTensor>(output_shape, data_type, 1, dst_q_info);

        matMul.configure(a.info(), b.info(), (_enable_bias) ? bias.info() : nullptr, dst.info(), matmul_info);
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

        // Fill tensors
        fill(CLAccessor(a), _hash + 1);
        fill(CLAccessor(b), _hash + 2);

        // Compute matMul kernel
        ITensorPack tensors_pack({ { ACL_SRC_0, &a },
            { ACL_SRC_1, &b },
            { ACL_DST, &dst }
        });

        if(_enable_bias)
        {
            // Allocate, fill and add bias to TensorPack obj
            bias.allocator()->allocate();
            if(is_quantized)
            {
                fill_bias_s32(CLAccessor(bias), _hash + 3, _min_bias, _max_bias);
            }
            else
            {
                fill(CLAccessor(bias), _hash + 3);
            }
            tensors_pack.add_tensor(ACL_SRC_2, &bias);
        }

        matMul.run(tensors_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &output_shape, bool pretranspose_a, bool pretranspose_b, DataType data_type,
                                      const QuantizationInfo &lhs_q_info, const QuantizationInfo &rhs_q_info, const QuantizationInfo &dst_q_info)
    {
        // We collapse dimensions > 3 onto dimension 3, i.e. 5D+ tensors will look like 4D
        // This is necessary unless we choose to extend gemm reference for 5D+ tensors
        TensorShape output_shape_collapsed = output_shape.collapsed_from(Window::DimZ);
        TensorShape shape_a_collapsed      = shape_a.collapsed_from(Window::DimZ);
        TensorShape shape_b_collapsed      = shape_b.collapsed_from(Window::DimZ);

        // Create reference
        SimpleTensor<T> a{ shape_a_collapsed, data_type, 1, lhs_q_info };
        SimpleTensor<T> b{ shape_b_collapsed, data_type, 1, rhs_q_info };
        SimpleTensor<T> c{ output_shape_collapsed, data_type, 1, dst_q_info };

        // Fill reference
        fill(a, _hash + 1);
        fill(b, _hash + 2);

        /* Note: Assuming the usual batch matmul dimensions A = (B x M x K), B = (B x K x N), if pretranspose_A is set to true, then A is assumed to be (B x K x M),
           therefore, A must be pre-transposed before passing it to the fixture. And, we transpose A again in the fixture to make it (B x M x K)
           in order to be able to call reference implementation that works with (B x M x K) input.
           Similarly, if pretranspose_B is set to true, then B is assumed to be (B x N x K), B must be pre-transposed before passing it to the fixture. */

        // Define transposed shapes
        TensorShape a_transposed_shape(a.shape());
        a_transposed_shape.set(0, a.shape().y());
        a_transposed_shape.set(1, a.shape().x());

        TensorShape b_transposed_shape(b.shape());
        b_transposed_shape.set(0, b.shape().y());
        b_transposed_shape.set(1, b.shape().x());

        // Define transposed tensors
        SimpleTensor<T> a_transposed{ a_transposed_shape, data_type };
        SimpleTensor<T> b_transposed{ b_transposed_shape, data_type };

        // pretranspose a if necessary
        if(pretranspose_a)
        {
            a_transposed = reference::permute<T>(a, PermutationVector(1U, 0U));
        }

        // pretranspose b if necessary
        if(pretranspose_b)
        {
            b_transposed = reference::permute<T>(b, PermutationVector(1U, 0U));
        }

        // Use transposed tensors if boolean enabled else use original tensors
        SimpleTensor<T> result = gemm_reference<T>((pretranspose_a) ? a_transposed : a, (pretranspose_b) ? b_transposed : b, c);

        // We reshape the gemm output back if the tensor is high dimensional
        if(output_shape_collapsed != output_shape)
        {
            result = reference::reshape_layer(result, output_shape);
        }

        return result;
    }

    template <typename U = T>
    typename std::enable_if < std::is_same<U, float>::value || std::is_same<U, half>::value, SimpleTensor<U >>::type gemm_reference(SimpleTensor<U> &a, SimpleTensor<U> &b, SimpleTensor<U> &c)
    {
        // Fill bias, then copy first dimension into subsequent dimensions to mimic broadcast
        // of bias tensor from shape [dst.dimension(0)] to [dst.tensor_shape()] in target kernel
        if(_enable_bias)
        {
            fill(c, _hash + 3);
            const int n          = c.shape().x();
            const int other_dims = c.shape().collapsed_from(1)[1];
            for(int i = 1; i < other_dims; ++i) // For all data, copy first n elements into remaining batches
            {
                memcpy(c.data() + i * n, c.data(), n * sizeof(T));
            }
        }
        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: alpha * A * B + 0 * C
        return reference::gemm<U>(a, b, c, 1.0f, (_enable_bias) ? 1.0f : 0.f);
    }

    template <typename U = T>
    typename std::enable_if < std::is_same<U, int8_t>::value || std::is_same<U, uint8_t>::value, SimpleTensor<U >>::type gemm_reference(SimpleTensor<U> &a, SimpleTensor<U> &b, SimpleTensor<U> &c)
    {
        const UniformQuantizationInfo aq = a.quantization_info().uniform();
        const UniformQuantizationInfo bq = b.quantization_info().uniform();
        const UniformQuantizationInfo cq = c.quantization_info().uniform();

        const SimpleTensor<int32_t> result = reference::gemmlowp_matrix_multiply_core<int32_t, U, U>(a, b, c.shape(), -aq.offset, -bq.offset);

        std::vector<int32_t> gemmlowp_multipliers{ 1 };
        std::vector<int32_t> gemmlowp_shifts{ 1 };
        const int            gemmlowp_offset = cq.offset;
        const float          scale           = aq.scale * bq.scale / cq.scale;

        quantization::calculate_quantized_multiplier(scale, &gemmlowp_multipliers[0], &gemmlowp_shifts[0]);
        constexpr int32_t gemmlowp_min_bound = std::numeric_limits<int32_t>::min();
        constexpr int32_t gemmlowp_max_bound = std::numeric_limits<int32_t>::max();

        SimpleTensor<int> bias{ c.shape(), DataType::S32 };
        if(_enable_bias)
        {
            // Identical to float implementation, fill and copy values of bias first dimension
            fill_bias_s32(bias, _hash + 3, _min_bias, _max_bias);
            const int          n          = bias.shape().x();
            const int          other_dims = bias.shape().collapsed_from(1)[1];
            const unsigned int dt_size    = sizeof(int32_t);
            for(int i = 1; i < other_dims; ++i)
            {
                memcpy(bias.data() + i * n, bias.data(), n * dt_size);
            }
        }
        else
        {
            fill_constant(bias, static_cast<int32_t>(0)); // effectively disable bias
        }

        const SimpleTensor<U> final_result = reference::gemmlowp_quantize_down_scale_by_fixedpoint<int32_t, U>(result, bias,
                                                                                                               gemmlowp_multipliers, gemmlowp_shifts, gemmlowp_offset, gemmlowp_min_bound, gemmlowp_max_bound);

        return final_result;
    }

    CLTensor        _target{};
    SimpleTensor<T> _reference{};
    bool            _enable_bias{ false };
    bool            _device_supports_export_to_cl_image{ true };
    bool            _device_supports_mmul{ true };
    int32_t         _min_bias{ 0 };
    int32_t         _max_bias{ 0 };
    int32_t         _hash{ 0 };
};

template <typename T, typename KernelType, bool use_mmul = false>
class MatMulKernelValidationFixture : public MatMulKernelGenericValidationFixture<T, KernelType, use_mmul>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape output_shape, bool pretranspose_a, bool pretranspose_b, int M0, int N0, int K0, bool export_rhs_to_cl_image, DataType data_type)
    {
        MatMulKernelGenericValidationFixture<T, KernelType, use_mmul>::setup(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, M0, N0, K0, export_rhs_to_cl_image, data_type,
                                                                             false /* enable bias */);
    }
};

template <typename T, typename KernelType, bool use_mmul = false>
class MatMulKernelWithBiasValidation : public MatMulKernelGenericValidationFixture<T, KernelType, use_mmul>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape output_shape, bool pretranspose_a, bool pretranspose_b, int M0, int N0, int K0, bool export_rhs_to_cl_image, DataType data_type)
    {
        MatMulKernelGenericValidationFixture<T, KernelType, use_mmul>::setup(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, M0, N0, K0, export_rhs_to_cl_image, data_type,
                                                                             true /* enable bias */);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_MATMULKERNELFIXTURE_H
