/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMM.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool disable_c = false, bool reinterpret_input_as_3d = false, bool reinterpret_ouput_as_3d = false>
class GEMMValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_c, TensorShape output_shape, float alpha, float beta, bool pretranspose, DataType data_type)
    {
        _target    = compute_target(shape_a, shape_b, shape_c, output_shape, alpha, beta, pretranspose, data_type);
        _reference = compute_reference(shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type);
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
                              bool pretranspose, DataType data_type)
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
        gemm.configure(&a, &b, (disable_c) ? nullptr : &c, &dst, alpha, beta, GEMMInfo(false, false, false, (reinterpret_ouput_as_3d ? output_shape[2] : 0), reinterpret_input_as_3d));
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

    SimpleTensor<T> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c, const TensorShape &output_shape, float alpha, float beta,
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
        SimpleTensor<T> c{ shape_c, data_type, 1 };

        // Fill reference
        fill(a, 0);
        fill(b, 1);
        if(!disable_c)
        {
            fill(c, 2);
            return reference::gemm<T>(a, b, c, alpha, beta);
        }
        else
        {
            // Setting beta to 0 will effectively disable C for the
            // computation of the reference: alpha * A * B + 0 * C
            return reference::gemm<T>(a, b, c, alpha, 0.f);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
class GEMMMatrixMultiplyReshapedValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0, bool interleave_lhs,
               bool interleave_rhs, DataType data_type, float alpha)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = false;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = true;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type, alpha);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha);
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

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, DataType data_type, float alpha)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs_reshaped, &rhs_reshaped, &dst, alpha, lhs_info, rhs_info, GEMMReshapeInfo(M, N, K));

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> c{ dst_shape, data_type, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemm<T>(lhs, rhs, c, alpha, 0.0f);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeLHSFunctionType, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
class GEMMMatrixMultiplyReshaped3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
               bool interleave_lhs,
               bool interleave_rhs, DataType data_type, float alpha)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave_lhs;
        lhs_info.transpose  = false;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = true;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type, alpha, m_h);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, m_h);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, DataType data_type, float alpha,
                              unsigned int m_h)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType lhs_reshaped;
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeLHSFunctionType reshape_lhs;
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_lhs.configure(&lhs, &lhs_reshaped, lhs_info);
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs_reshaped, &rhs_reshaped, &dst, alpha, lhs_info, rhs_info, GEMMReshapeInfo(M, N, K, 1, 1, m_h));

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!lhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        reshape_lhs.run();
        reshape_rhs.run();
        gemm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, unsigned int m_h)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> c{ dst_shape, data_type, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemm<T>(lhs, rhs, c, alpha, 0.0f);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
class GEMMMatrixMultiplyReshapedOnlyRHSValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int h0,
               bool interleave_rhs, bool transpose_rhs, DataType data_type, float alpha)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type, alpha);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha);
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

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, DataType data_type, float alpha)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs, &rhs_reshaped, &dst, alpha, lhs_info, rhs_info, GEMMReshapeInfo(M, N, K));

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        reshape_rhs.run();
        gemm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape[0]          = rhs_shape[0];
        dst_shape[1]          = lhs_shape[1];

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> c{ dst_shape, data_type, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemm<T>(lhs, rhs, c, alpha, 0.0f);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename T, typename ReshapeRHSFunctionType, typename GEMMFunctionType>
class GEMMMatrixMultiplyReshapedOnlyRHS3DValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int m_w, unsigned int m_h, unsigned int n, unsigned int k, unsigned int batch_size, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int h0,
               bool interleave_rhs, bool transpose_rhs, DataType data_type, float alpha)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = m0;
        lhs_info.k0 = k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = n0;
        rhs_info.k0         = k0;
        rhs_info.h0         = h0;
        rhs_info.interleave = interleave_rhs;
        rhs_info.transpose  = transpose_rhs;

        // In case of GEMM3D, m is the product between m_w and m_h
        const unsigned int m = m_w * m_h;

        // Set the tensor shapes for LHS and RHS matrices
        const TensorShape lhs_shape(k, m, batch_size);
        const TensorShape rhs_shape(n, k, batch_size);

        _target    = compute_target(lhs_shape, rhs_shape, lhs_info, rhs_info, data_type, alpha, m_h);
        _reference = compute_reference(lhs_shape, rhs_shape, data_type, alpha, m_h);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &lhs_shape, const TensorShape &rhs_shape, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, DataType data_type, float alpha,
                              unsigned int m_h)
    {
        // Create tensors
        TensorType lhs = create_tensor<TensorType>(lhs_shape, data_type, 1);
        TensorType rhs = create_tensor<TensorType>(rhs_shape, data_type, 1);
        TensorType rhs_reshaped;
        TensorType dst;

        const unsigned int M = lhs_shape[1];
        const unsigned int N = rhs_shape[0];
        const unsigned int K = lhs_shape[0];

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        ReshapeRHSFunctionType reshape_rhs;
        GEMMFunctionType       gemm;
        reshape_rhs.configure(&rhs, &rhs_reshaped, rhs_info);
        gemm.configure(&lhs, &rhs_reshaped, &dst, alpha, lhs_info, rhs_info, GEMMReshapeInfo(M, N, K, 1, 1, m_h));

        ARM_COMPUTE_EXPECT(lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(rhs.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!lhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!rhs_reshaped.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(lhs), 0);
        fill(AccessorType(rhs), 1);

        // Compute GEMM
        reshape_rhs.run();
        gemm.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &lhs_shape, const TensorShape &rhs_shape, DataType data_type, float alpha, unsigned int m_h)
    {
        TensorShape dst_shape = lhs_shape;
        dst_shape.set(0, rhs_shape[0]);
        dst_shape.set(1, lhs_shape[1] / m_h);
        dst_shape.set(2, m_h);
        dst_shape.set(3, lhs_shape[2]);

        // Create reference
        SimpleTensor<T> lhs{ lhs_shape, data_type, 1 };
        SimpleTensor<T> rhs{ rhs_shape, data_type, 1 };
        SimpleTensor<T> c{ dst_shape, data_type, 1 };

        // Fill reference
        fill(lhs, 0);
        fill(rhs, 1);

        return reference::gemm<T>(lhs, rhs, c, alpha, 0.0f);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMM_FIXTURE */
