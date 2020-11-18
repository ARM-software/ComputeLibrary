/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/GEMM.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool disable_c = false, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false>
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
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(lo, hi);
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
                       GEMMInfo(false, false, false, (reinterpret_output_as_3d ? output_shape[2] : 0), reinterpret_input_as_3d, false, GEMMLowpOutputStageInfo(), false, (reinterpret_input_as_3d
                                || reinterpret_output_as_3d)));
        ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!c.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

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

        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: alpha * A * B + 0 * C
        return reference::gemm<T>(a, b, c, alpha, disable_c ? 0.f : beta);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        std::uniform_real_distribution<> distribution_inf(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
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
        GEMMFunctionType gemm;
        gemm.configure(gpu_arch, &lhs, &rhs, &bias, &dst, alpha, beta, false, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
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
        GEMMFunctionType gemm;
        gemm.configure(gpu_arch, &lhs, &rhs, &bias, &dst, alpha, beta, false, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        std::uniform_real_distribution<> distribution_inf(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
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
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(gpu_arch, &lhs_reshaped, &rhs_reshaped, &bias, &dst, alpha, beta, true, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
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
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(gpu_arch, &lhs_reshaped, &rhs_reshaped, &bias, &dst, alpha, beta, true, reshape_info, fp16_mixed_precision, act_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType, bool fp_mixed_precision = false>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        std::uniform_real_distribution<> distribution_inf(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
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
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs_reshaped, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType, bool fp_mixed_precision = false>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
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
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs_reshaped, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        std::uniform_real_distribution<> distribution_inf(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
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
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
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
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;

        validate_result = bool(reshape_rhs.validate(rhs.info(), rhs_reshaped.info(), rhs_info));
        validate_result = validate_result || !rhs_info.export_to_cl_image;
        if(!validate_result)
        {
            return nullptr;
        }

        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        if(has_pad_y)
        {
            // Add dummy padding into lhs to validate has_pad_y path
            lhs.info()->extend_padding(PaddingSize(2, 0, 2, 0));
            dst.info()->extend_padding(PaddingSize(2, 0, 1, 0));
        }

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        reshape_rhs.run();
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);

        // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
        std::uniform_real_distribution<> distribution_inf(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
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
        GEMMFunctionType gemm;
        gemm.configure(&lhs, &rhs, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        gemm.run();

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

template <typename TensorType, typename AccessorType, typename T, typename GEMMFunctionType>
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
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
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
        GEMMFunctionType gemm;
        gemm.configure(&lhs, &rhs, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);
        fill(AccessorType(bias), 2);

        // Compute GEMM
        gemm.run();

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
