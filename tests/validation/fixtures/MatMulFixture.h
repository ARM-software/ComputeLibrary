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
#ifndef TESTS_VALIDATION_FIXTURES_MATMULFIXTURE
#define TESTS_VALIDATION_FIXTURES_MATMULFIXTURE

#include "arm_compute/core/Types.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/ReshapeLayer.h"
#include <random>
namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class MatMulValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape output_shape, bool pretranspose_a, bool pretranspose_b, DataType data_type)
    {
        // For brevity, the input shapes are assumed to be not-transposed for both Lhs and Rhs matrices.
        if(pretranspose_a)
        {
            permute(shape_a, PermutationVector(1U, 0U));
        }
        if(pretranspose_b)
        {
            permute(shape_b, PermutationVector(1U, 0U));
        }
        _target    = compute_target(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, data_type);
        _reference = compute_reference(shape_a, shape_b, output_shape, pretranspose_a, pretranspose_b, data_type);
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
    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &output_shape, bool pretranspose_a, bool pretranspose_b, DataType data_type)
    {
        // 1. Create Classes and configure function
        // Create tensors
        TensorType   a   = create_tensor<TensorType>(shape_a, data_type, 1);
        TensorType   b   = create_tensor<TensorType>(shape_b, data_type, 1);
        TensorType   dst = create_tensor<TensorType>(output_shape, data_type, 1);
        FunctionType matmul;
        // Configure MatMulInfo class
        MatMulInfo info;
        info.adj_lhs(pretranspose_a);
        info.adj_rhs(pretranspose_b);
        matmul.configure(&a, &b, &dst, info);
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

        // 2. Fill tensors and run once
        // Fill tensors
        fill(AccessorType(a), 0);
        fill(AccessorType(b), 1);
        matmul.run(); // First run

        return dst;
    }
    SimpleTensor<T> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &output_shape, bool pretranspose_a, bool pretranspose_b, DataType data_type)
    {
        // We collapse dimensions > 3 onto dimension 3, i.e. 5D+ tensors will look like 4D
        // This is necessary unless we choose to extend gemm reference for 5D+ tensors
        TensorShape output_shape_collapsed = output_shape.collapsed_from(Window::DimW);
        TensorShape a_shape_collapsed      = shape_a.collapsed_from(Window::DimW);
        TensorShape b_shape_collapsed      = shape_b.collapsed_from(Window::DimW);

        // Create reference
        SimpleTensor<T> a{ a_shape_collapsed, data_type, 1 };
        SimpleTensor<T> b{ b_shape_collapsed, data_type, 1 };
        SimpleTensor<T> c{ output_shape_collapsed, data_type, 1 };

        // Fill reference
        fill(a, 0);
        fill(b, 1);

        /* Note: Assuming the usual batch matmul dimensions A = (B x M x K), B = (B x K x N), if pretranspose_a is set to true, then A is assumed to be (B x K x M),
           therefore, A must be pre-transposed before passing it to the fixture. And, we transpose A again in the fixture to make it (B x M x K)
           in order to be able to call reference implementation that works with (B x M x K) input.
           Similarly, if pretranspose_b is set to true, then B is assumed to be (B x N x K), B must be pre-transposed before passing it to the fixture. */

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

        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: alpha * A * B + 0 * C
        // Use transposed tensors if boolean enabled else use original tensors
        SimpleTensor<T> result = reference::gemm<T>((pretranspose_a) ? a_transposed : a, (pretranspose_b) ? b_transposed : b, c, 1.0f, 0.f);

        // We reshape the gemm output back if the tensor is high dimensional
        if(output_shape_collapsed != output_shape)
        {
            result = reference::reshape_layer(result, output_shape);
        }

        return result;
    }
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* TESTS_VALIDATION_FIXTURES_MATMULFIXTURE */
