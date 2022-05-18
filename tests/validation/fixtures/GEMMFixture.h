/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_GEMM_FIXTURE
#define ARM_COMPUTE_TEST_GEMM_FIXTURE

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "src/core/experimental/PostOpUtils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/PostOps.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool disable_c = false, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, bool pretranspose_a = false, bool pretranspose_b = false>
class GEMMValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_c, TensorShape output_shape, float alpha, float beta, bool pretranspose, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(pretranspose);
        _target    = compute_target(shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type);
        _reference = compute_reference(shape_a, shape_b, output_shape, alpha, beta, data_type);
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

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c, const TensorShape &output_shape, float alpha, float beta,
                              DataType data_type)
    {
        // Create tensors
        TensorType a   = create_tensor<TensorType>(shape_a, data_type, 1);
        TensorType b   = create_tensor<TensorType>(shape_b, data_type, 1);
        TensorType c   = create_tensor<TensorType>(shape_c, data_type, 1);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1);

        // Create and configure function
        FunctionType gemm;
        // The GEMMinfo includes the values of the depth in case of reinterpreted 3d output.
        // If the output shape has the same number of dimensions of the input the method called is a 2D matrix multiplication (depth_output_reinterpreted_as_3D = 0),
        // in the other case we have to use the reinterpreted version of GEMM (depth_output_reinterpreted_as_3D = depth of the 3D output).
        gemm.configure(&a,
                       &b,
                       (disable_c) ? nullptr : &c,
                       &dst,
                       alpha, beta,
                       GEMMInfo(false, false, false, (reinterpret_output_as_3d ? output_shape[2] : 0), reinterpret_input_as_3d, false, GEMMLowpOutputStageInfo(), false, false, (reinterpret_input_as_3d
                                || reinterpret_output_as_3d)));
        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(c.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &a, &b, &c, &dst });

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!c.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(a), 0);
        fill(AccessorType(b), 1);
        if(!disable_c)
        {
            fill(AccessorType(c), 2);
        }

        // Compute GEMM function
        gemm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &output_shape, float alpha, float beta,
                                      DataType data_type)
    {
        TensorShape shape_a_to_use = shape_a;

        if(reinterpret_input_as_3d)
        {
            // Collapse the second and third dimension if the input is 3D
            shape_a_to_use.collapse(2U, 1U);
        }

        // Create reference
        SimpleTensor<T> a{ shape_a_to_use, data_type, 1 };
        SimpleTensor<T> b{ shape_b, data_type, 1 };
        SimpleTensor<T> c{ output_shape, data_type, 1 };

        // Fill reference
        fill(a, 0);
        fill(b, 1);
        fill(c, 2);

        if(reinterpret_input_as_3d || reinterpret_output_as_3d)
        {
            const int n          = shape_b[0];
            const int m          = reinterpret_output_as_3d ? output_shape[1] * output_shape[2] : output_shape[1];
            const int batch_size = reinterpret_output_as_3d ? output_shape[3] : output_shape[2];

            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(c.data() + i * n, c.data(), n * sizeof(T));
            }
        }
        
        /* Note: Assuming the usual batch matmul dimensions A = (B x M x K), B = (B x K x N), if pretranspose_A is set to true, then A is assumed to be (B x K x M),
           therefore, A must be pre-transposed before passing it to the fixture. And, we transpose A again in the fixture to make it (B x M x K)
           in order to be able to call reference implementation that works with (B x M x K) input.
           Similarly, if pretranspose_B is set to true, then B is assumed to be (B x N x K), B must be pre-transposed before passing it to the fixture. */
           
        // Define transposed shapes
        TensorShape a_transposed_shape(a.shape().y(), a.shape().x());
        TensorShape b_transposed_shape(b.shape().y(), b.shape().x());

        // Define transposed tensors
        SimpleTensor<T> a_transposed{ a_transposed_shape, data_type };
        SimpleTensor<T> b_transposed{ b_transposed_shape, data_type };

        // pretranspose a if necessary
        if(pretranspose_a)
        {
            transpose_matrix<T>(a, a_transposed);
        }

        // pretranspose b if necessary
        if(pretranspose_b)
        {
            transpose_matrix<T>(b, b_transposed);
        }

        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: alpha * A * B + 0 * C
        // Use transposed tensors if boolean enabled else use original tensors
        return reference::gemm<T>((pretranspose_a) ? a_transposed : a, (pretranspose_b) ? b_transposed : b, c, alpha, disable_c ? 0.f : beta);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMOperatorType>
class GEMMMatrixMultiplyValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, float alpha, float beta, bool broadcast_bias, bool fp16_mixed_precision, const ActivationLayerInfo &act_info,
               DataType data_type, GPUTarget gpu_arch)
    {
        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, data_type, alpha, beta, broadcast_bias, fp16_mixed_precision, act_info, gpu_arch);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                              bool fp16_mixed_precision, const ActivationLayerInfo &act_info, GPUTarget gpu_arch)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst;

        const unsigned int m = lhs_shape[1];
        const unsigned int n = rhs_shape[0];
        const unsigned int k = lhs_shape[0];
        GEMMReshapeInfo    reshape_info(m, n, k, 1, 1, 0, false, broadcast_bias);

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        GEMMOperatorType gemm;
        gemm.configure(gpu_arch, lhs.info(), rhs.info(), bias.info(), dst.info(), alpha, beta, false, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &bias, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMOperatorType>
class GEMMMatrixMultiply3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, float alpha, float beta, bool broadcast_bias, bool fp16_mixed_precision,
               const ActivationLayerInfo &act_info, DataType data_type, GPUTarget gpu_arch)
    {
        ARM_COMPUTE_UNUSED(broadcast_bias);

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n, 1, 1);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, data_type, alpha, beta, m_h, fp16_mixed_precision, act_info, gpu_arch);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, m_h, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                              bool fp16_mixed_precision, const ActivationLayerInfo &act_info, GPUTarget gpu_arch)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst;

        const unsigned int m = lhs_shape[1];
        const unsigned int n = rhs_shape[0];
        const unsigned int k = lhs_shape[0];
        GEMMReshapeInfo    reshape_info(m, n, k, 1, 1, m_h, false, true);

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        GEMMOperatorType gemm;
        gemm.configure(gpu_arch, lhs.info(), rhs.info(), bias.info(), dst.info(), alpha, beta, false, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &bias, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        // In case of broadcast, we need simply copy the first into the following "M" ones
        for(int i = 1; i < m * batch_size; i++)
        {
            memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMOperatorType>
class GEMMMatrixMultiplyInterleavedTransposedValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, float alpha, float beta, unsigned int v0, unsigned int h0, bool broadcast_bias, bool fp16_mixed_precision,
               const ActivationLayerInfo &act_info, DataType data_type, GPUTarget gpu_arch)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = 4;
        lhs_info.k0         = 4;
        lhs_info.v0         = v0;
        lhs_info.interleave = true;
        lhs_info.transpose  = true;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = 16 / sizeof(T);
        rhs_info.k0         = 1;
        rhs_info.h0         = h0;
        rhs_info.interleave = false;
        rhs_info.transpose  = false;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, fp16_mixed_precision, act_info, gpu_arch);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, bool fp16_mixed_precision, const ActivationLayerInfo &act_info, GPUTarget gpu_arch)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int m = lhs_shape[1];
        const unsigned int n = rhs_shape[0];
        const unsigned int k = lhs_shape[0];
        GEMMReshapeInfo    reshape_info(m, n, k, rhs_info.h0, lhs_info.v0, 0, false, broadcast_bias);

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;
        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(gpu_arch, lhs_reshaped.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, true, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMOperatorType>
class GEMMMatrixMultiplyInterleavedTransposed3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, float alpha, float beta, unsigned int v0, unsigned int h0, bool broadcast_bias,
               bool fp16_mixed_precision, const ActivationLayerInfo &act_info, DataType data_type, GPUTarget gpu_arch)
    {
        ARM_COMPUTE_UNUSED(broadcast_bias);

        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = 4;
        lhs_info.k0         = 4;
        lhs_info.v0         = v0;
        lhs_info.interleave = true;
        lhs_info.transpose  = true;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = 16 / sizeof(T);
        rhs_info.k0         = 1;
        rhs_info.h0         = h0;
        rhs_info.interleave = false;
        rhs_info.transpose  = false;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n, 1, 1);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, m_h, fp16_mixed_precision, act_info, gpu_arch);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, m_h, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, unsigned int m_h, bool fp16_mixed_precision, const ActivationLayerInfo &act_info, GPUTarget gpu_arch)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int m = lhs_shape[1];
        const unsigned int n = rhs_shape[0];
        const unsigned int k = lhs_shape[0];
        GEMMReshapeInfo    reshape_info(m, n, k, rhs_info.h0, lhs_info.v0, m_h, false, true);

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;
        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(gpu_arch, lhs_reshaped.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, true, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        // In case of broadcast, we need simply copy the first into the following "M" ones
        for(int i = 1; i < m * batch_size; i++)
        {
            memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMOperatorType, bool fp_mixed_precision = false>
class GEMMMatrixMultiplyReshapedValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0, bool interleave_lhs,
               bool interleave_rhs, bool export_to_cl_image, DataType data_type, float alpha, float beta, bool broadcast_bias, bool lhs_transpose, const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = lhs_transpose;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = !lhs_transpose;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;
        kernel_info.fp_mixed_precision      = fp_mixed_precision;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        if(fp_mixed_precision)
        {
            return reference::activation_layer(reference::gemm_mixed_precision<T>(lhs, rhs, bias, alpha, beta), act_info);
        }
        else
        {
            return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
        }
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

/** (EXPERIMENTAL_POST_OPS)*/
template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMOperatorType, bool fp_mixed_precision = false>
class GEMMMatrixMultiplyReshapedWithPostOpsValidationFixture : public framework::Fixture
{
public:
    using PostOpArgBroadcast = std::tuple<bool, bool, bool>; // Instruct fixture if we need broadcasting in dimension 0, 1, 2 of each PostOp argument
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0, bool interleave_lhs,
               bool interleave_rhs, bool export_to_cl_image, DataType data_type, float alpha, float beta, bool broadcast_bias, bool lhs_transpose, const ActivationLayerInfo &act_info,
               const experimental::PostOpList<PostOpArgBroadcast> &post_ops)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = lhs_transpose;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = !lhs_transpose;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);
        auto post_ops_with_shapes = experimental::transform_post_op_list_arguments<PostOpArgBroadcast, TensorShape>(post_ops,
                                                                                                                    [ = ](auto broadcast)
        {
            return TensorShape
            {
                std::get<0>(broadcast) ? 1 : n,
                std::get<1>(broadcast) ? 1 : m,
                std::get<2>(broadcast) ? 1 : batch_size,
            };
        });

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);

        // Create post op tensors and populate post op with them
        std::vector<TensorType> post_op_tensors_holder{};
        auto                    populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, ITensorInfo *>(post_ops,
                                                                                                                                [&post_op_tensors_holder, &data_type](auto shape)
        {
            auto t = create_tensor<TensorType>(shape, data_type, 1);
            post_op_tensors_holder.push_back(std::move(t));
            return post_op_tensors_holder.back().info();
        });
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;
        kernel_info.fp_mixed_precision      = fp_mixed_precision;
        kernel_info.post_ops                = populated_post_ops;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(tensor.info()->is_resizable());
        }

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &bias, &dst });
            for(auto &tensor : post_op_tensors_holder)
            {
                add_padding_x({ &tensor });
            }
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        for(auto &tensor : post_op_tensors_holder)
        {
            tensor.allocator()->allocate();
        }

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(!tensor.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            fill(AccessorType(post_op_tensors_holder.at(i)), 3 + i);
        }

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            gemm_pack.add_tensor(experimental::get_post_op_arg_type(i), &post_op_tensors_holder.at(i));
        }
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };
        // Create post op tensors and populate post op with them
        auto populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, SimpleTensor<T>>(post_ops, [&data_type](auto shape)
        {
            return SimpleTensor<T> { shape, data_type, 1 };
        });

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        int tensor_idx = 0;
        fill(lhs, tensor_idx++);
        fill(rhs, tensor_idx++);
        fill(bias, tensor_idx++);
        for(auto &op : populated_post_ops.get_list())
        {
            for(auto tensor : op->arguments())
            {
                fill(*tensor, tensor_idx++);
            }
        }

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        SimpleTensor<T> out;
        if(fp_mixed_precision)
        {
            out = reference::gemm_mixed_precision<T>(lhs, rhs, bias, alpha, beta);
        }
        else
        {
            out = reference::gemm<T>(lhs, rhs, bias, alpha, beta);
        }
        // Ignore activation info if post ops are used instead
        if(populated_post_ops.size() > 0)
        {
            out = reference::post_ops<T>(out, populated_post_ops);
        }
        else
        {
            out = reference::activation_layer(out, act_info);
        }
        return out;
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSOperatorType, typename ReshapeRHSOperatorType, typename GEMMOperatorType, bool fp_mixed_precision = false>
class GEMMMatrixMultiplyReshaped3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
               bool interleave_lhs, bool interleave_rhs, bool export_to_cl_image, DataType data_type, float alpha, float beta, bool lhs_transpose, const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = lhs_transpose;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = !lhs_transpose;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n, 1, 1);

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, m_h, act_info);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, m_h, act_info);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, unsigned int m_h, const ActivationLayerInfo &act_info)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = m_h;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = true;
        kernel_info.activation_info         = act_info;
        kernel_info.fp_mixed_precision      = fp_mixed_precision;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSOperatorType reshape_lhs;
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);
        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &lhs_reshaped, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!lhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_lhs_pack = { { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } };
        reshape_lhs.run(reshape_lhs_pack);
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        // In case of broadcast, we need simply copy the first into the following "M" ones
        for(int i = 1; i < m * batch_size; i++)
        {
            memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
        }

        if(fp_mixed_precision)
        {
            return reference::activation_layer(reference::gemm_mixed_precision<T>(lhs, rhs, bias, alpha, beta), act_info);
        }
        else
        {
            return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
        }
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSOperatorType, typename GEMMOperatorType>
class GEMMMatrixMultiplyReshapedOnlyRHSValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int h0,
               bool interleave_rhs, bool transpose_rhs, bool export_to_cl_image, DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = transpose_rhs;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

/** (EXPERIMENTAL_POST_OPS)*/
template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSOperatorType, typename GEMMOperatorType>
class GEMMMatrixMultiplyReshapedOnlyRHSWithPostOpsValidationFixture : public framework::Fixture
{
public:
    using PostOpArgBroadcast = std::tuple<bool, bool, bool>; // Instruct fixture if we need broadcasting in dimension 0, 1, 2 of each PostOp argument
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int h0,
               bool interleave_rhs, bool transpose_rhs, bool export_to_cl_image, DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info,
               const experimental::PostOpList<PostOpArgBroadcast> &post_ops)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = transpose_rhs;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);
        auto post_ops_with_shapes = experimental::transform_post_op_list_arguments<PostOpArgBroadcast, TensorShape>(post_ops,
                                                                                                                    [ = ](auto broadcast)
        {
            return TensorShape
            {
                std::get<0>(broadcast) ? 1 : n,
                std::get<1>(broadcast) ? 1 : m,
                std::get<2>(broadcast) ? 1 : batch_size,
            };
        });

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;
        // Create post op tensors and populate post op with them
        std::vector<TensorType> post_op_tensors_holder{};
        auto                    populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, ITensorInfo *>(post_ops,
                                                                                                                                [&post_op_tensors_holder, &data_type](auto shape)
        {
            auto t = create_tensor<TensorType>(shape, data_type, 1);
            post_op_tensors_holder.push_back(std::move(t));
            return post_op_tensors_holder.back().info();
        });

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;
        kernel_info.post_ops                = populated_post_ops;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(tensor.info()->is_resizable());
        }

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &rhs_reshaped, &bias, &dst });
            for(auto &tensor : post_op_tensors_holder)
            {
                add_padding_x({ &tensor });
            }
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        for(auto &tensor : post_op_tensors_holder)
        {
            tensor.allocator()->allocate();
        }

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(!tensor.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            fill(AccessorType(post_op_tensors_holder.at(i)), 3 + i);
        }

        // Compute GEMM
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            gemm_pack.add_tensor(experimental::get_post_op_arg_type(i), &post_op_tensors_holder.at(i));
        }
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };
        // Create post op tensors and populate post op with them
        auto populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, SimpleTensor<T>>(post_ops, [&data_type](auto shape)
        {
            return SimpleTensor<T> { shape, data_type, 1 };
        });

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        int tensor_idx = 0;
        fill(lhs, tensor_idx++);
        fill(rhs, tensor_idx++);
        fill(bias, tensor_idx++);
        for(auto &op : populated_post_ops.get_list())
        {
            for(auto tensor : op->arguments())
            {
                fill(*tensor, tensor_idx++);
            }
        }

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        SimpleTensor<T> out;
        out = reference::gemm<T>(lhs, rhs, bias, alpha, beta);
        // Ignore activation info if post ops are used instead
        if(populated_post_ops.size() > 0)
        {
            out = reference::post_ops<T>(out, populated_post_ops);
        }
        else
        {
            out = reference::activation_layer(out, act_info);
        }
        return out;
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSOperatorType, typename GEMMOperatorType>
class GEMMMatrixMultiplyReshapedOnlyRHS3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int h0,
               bool interleave_rhs, bool transpose_rhs, bool export_to_cl_image, bool has_pad_y, DataType data_type, float alpha, float beta, const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = n0;
        rhs_info.k0                 = k0;
        rhs_info.h0                 = h0;
        rhs_info.interleave         = interleave_rhs;
        rhs_info.transpose          = transpose_rhs;
        rhs_info.export_to_cl_image = export_to_cl_image;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n, 1, 1);

        _target = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, m_h, act_info, has_pad_y);
        if(validate_result)
        {
            _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, m_h, act_info);
        }
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta,
                              unsigned int m_h, const ActivationLayerInfo &act_info, bool has_pad_y)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = m_h;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = true;
        kernel_info.activation_info         = act_info;
        kernel_info.has_pad_y               = has_pad_y;

        // The output tensor will be auto-initialized within the function
        // Create and configure function
        ReshapeRHSOperatorType reshape_rhs;
        GEMMOperatorType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_rhs.configure(rhs.info(), rhs_reshaped.info(), rhs_info);
        gemm.configure(lhs.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        if(has_pad_y)
        {
            // Add dummy padding into lhs to validate has_pad_y path
            lhs.info()->extend_padding(PaddingSize(2, 0, 2, 0));
            dst.info()->extend_padding(PaddingSize(2, 0, 1, 0));
        }

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        // We do not pad when using image as it needs to comply to strict pitch alignment restrictions
        if(!rhs_info.export_to_cl_image)
        {
            add_padding_x({ &lhs, &rhs, &rhs_reshaped, &bias, &dst });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs_reshaped.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack reshape_rhs_pack = { { ACL_SRC, &rhs }, { ACL_DST, &rhs_reshaped } };
        reshape_rhs.run(reshape_rhs_pack);
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        // In case of broadcast, we need simply copy the first into the following "M" ones
        for(int i = 1; i < m * batch_size; i++)
        {
            memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    bool            validate_result = true;
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMOperatorType>
class GEMMMatrixMultiplyNativeValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, DataType data_type, float alpha, float beta, bool broadcast_bias,
               const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0 = n0;
        rhs_info.k0 = k0;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;

        // Create and configure function
        GEMMOperatorType gemm;
        gemm.configure(lhs.info(), rhs.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &bias, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMOperatorType>
class GEMMMatrixMultiplyNativeWithPostOpsValidationFixture : public framework::Fixture
{
public:
    using PostOpArgBroadcast = std::tuple<bool, bool, bool>; // Instruct fixture if we need broadcasting in dimension 0, 1, 2 of each PostOp argument
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, DataType data_type, float alpha, float beta, bool broadcast_bias,
               const ActivationLayerInfo &act_info, const experimental::PostOpList<PostOpArgBroadcast> &post_ops)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0 = n0;
        rhs_info.k0 = k0;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n,
                                     broadcast_bias ? 1 : m,
                                     broadcast_bias ? 1 : batch_size);
        const auto post_ops_with_shapes = experimental::transform_post_op_list_arguments<PostOpArgBroadcast, TensorShape>(post_ops,
                                                                                                                          [ = ](auto broadcast)
        {
            return TensorShape
            {
                std::get<0>(broadcast) ? 1 : n,
                std::get<1>(broadcast) ? 1 : m,
                std::get<2>(broadcast) ? 1 : batch_size,
            };
        });

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, broadcast_bias, act_info, post_ops_with_shapes);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
        library->fill_borders_with_garbage(tensor, distribution_inf, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, bool broadcast_bias, const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst;
        // Create post op tensors and populate post op with them
        std::vector<TensorType> post_op_tensors_holder{};
        auto                    populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, ITensorInfo *>(post_ops,
                                                                                                                                [&post_op_tensors_holder, &data_type](auto shape)
        {
            auto t = create_tensor<TensorType>(shape, data_type, 1);
            post_op_tensors_holder.push_back(std::move(t));
            return post_op_tensors_holder.back().info();
        });

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = broadcast_bias;
        kernel_info.activation_info         = act_info;
        kernel_info.post_ops                = populated_post_ops;

        // Create and configure function
        GEMMOperatorType gemm;
        gemm.configure(lhs.info(), rhs.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(tensor.info()->is_resizable());
        }

        add_padding_x({ &lhs, &rhs, &bias, &dst });
        for(auto &tensor : post_op_tensors_holder)
        {
            add_padding_x({ &tensor });
        }

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        for(auto &tensor : post_op_tensors_holder)
        {
            tensor.allocator()->allocate();
        }

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        for(const auto &tensor : post_op_tensors_holder)
        {
            ARM_COMPUTE_ASSERT(!tensor.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            fill(AccessorType(post_op_tensors_holder.at(i)), 3 + i);
        }

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        for(size_t i = 0; i < post_op_tensors_holder.size(); ++i)
        {
            gemm_pack.add_tensor(experimental::get_post_op_arg_type(i), &post_op_tensors_holder.at(i));
        }
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, bool broadcast_bias,
                                      const ActivationLayerInfo &act_info, const experimental::PostOpList<TensorShape> &post_ops)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };
        // Create post op tensors and populate post op with them
        auto populated_post_ops = experimental::transform_post_op_list_arguments<TensorShape, SimpleTensor<T>>(post_ops, [&data_type](auto shape)
        {
            return SimpleTensor<T> { shape, data_type, 1 };
        });

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        int tensor_idx = 0;
        fill(lhs, tensor_idx++);
        fill(rhs, tensor_idx++);
        fill(bias, tensor_idx++);
        for(auto &op : populated_post_ops.get_list())
        {
            for(auto tensor : op->arguments())
            {
                fill(*tensor, tensor_idx++);
            }
        }

        if(broadcast_bias)
        {
            // In case of broadcast, we need simply copy the first into the following "M" ones
            for(int i = 1; i < m * batch_size; i++)
            {
                memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
            }
        }

        SimpleTensor<T> out;
        out = reference::gemm<T>(lhs, rhs, bias, alpha, beta);
        // Ignore activation info if post ops are used instead
        if(populated_post_ops.size() > 0)
        {
            out = reference::post_ops<T>(out, populated_post_ops);
        }
        else
        {
            out = reference::activation_layer(out, act_info);
        }
        return out;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMOperatorType>
class GEMMMatrixMultiplyNative3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, DataType data_type, float alpha, float beta,
               const ActivationLayerInfo &act_info)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0 = n0;
        rhs_info.k0 = k0;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);
        const TensorShape bias_shape(n, 1, 1);

        _target    = compute_target(lhs_shape, rhs_shape, bias_shape, lhs_info, rhs_info, data_type, alpha, beta, m_h, act_info);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, beta, m_h, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const TensorShape &bias_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                              DataType data_type, float alpha, float beta, unsigned int m_h, const ActivationLayerInfo &act_info)
    {
        // Create tensors
        TensorType lhs  = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs  = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType bias = create_tensor<TensorType>(bias_shape, data_type, 1);
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];
        GEMMKernelInfo     kernel_info;
        kernel_info.m                       = M;
        kernel_info.n                       = N;
        kernel_info.k                       = K;
        kernel_info.depth_output_gemm3d     = m_h;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = true;
        kernel_info.activation_info         = act_info;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        GEMMOperatorType gemm;
        gemm.configure(lhs.info(), rhs.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_ASSERT(lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());

        add_padding_x({ &lhs, &rhs, &bias, &dst });

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, float beta, unsigned int m_h,
                                      const ActivationLayerInfo &act_info)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> bias{ dst_shape, data_type, 1 };

        const int n          = rhs_shape[0];
        const int m          = lhs_shape[1];
        const int batch_size = lhs_shape[2];

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);
        fill(bias, 2);

        // In case of broadcast, we need simply copy the first into the following "M" ones
        for(int i = 1; i < m * batch_size; i++)
        {
            memcpy(bias.data() + i * n, bias.data(), n * sizeof(T));
        }

        return reference::activation_layer(reference::gemm<T>(lhs, rhs, bias, alpha, beta), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMM_FIXTURE */
