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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_MATMULKERNELFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_MATMULKERNELFIXTURE_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/MatMulAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuMatMul.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/ReshapeLayer.h"
#include "tests/validation/Validation.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename U>
void fill(U &&tensor, int i)
{
    switch (tensor.data_type())
    {
        case DataType::F16:
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{-1.0f, 1.0f};
            library->fill(tensor, distribution, i);
            break;
        }
        case DataType::F32:
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
            break;
        }
        default:
            library->fill_tensor_uniform(tensor, i);
    }
}

} // namespace
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuMatMulValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape lhs_shape,
               TensorShape rhs_shape,
               TensorShape output_shape,
               bool        transpose_a,
               bool        transpose_b,
               int         M0,
               int         N0,
               int         K0,
               bool        export_rhs_to_cl_image,
               DataType    data_type)
    {
        //For brevity, the input shapes are assumed to be not-transposed for both a and b matrices.
        if (transpose_a)
        {
            permute(lhs_shape, PermutationVector(1U, 0U));
        }
        if (transpose_b)
        {
            permute(rhs_shape, PermutationVector(1U, 0U));
        }

        // Skip configurations unsupported by the device.
        _device_supports_export_to_cl_image = image2d_from_buffer_supported(CLKernelLibrary::get().get_device());
        if (!_device_supports_export_to_cl_image && export_rhs_to_cl_image)
        {
            ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
            framework::ARM_COMPUTE_PRINT_INFO();
            return; // Note: Also need to skip the validate in corresponding FIXTURE_DATA_TEST_CASEs.
        }

        _target    = compute_target(lhs_shape, rhs_shape, transpose_a, transpose_b, M0, N0, K0, export_rhs_to_cl_image,
                                    data_type);
        _reference = compute_reference(lhs_shape, rhs_shape, output_shape, transpose_a, transpose_b, data_type);
    }

protected:
    TensorType compute_target(TensorShape &shape_a,
                              TensorShape &shape_b,
                              bool         transpose_a,
                              bool         transpose_b,
                              int          M0,
                              int          N0,
                              int          K0,
                              bool         export_rhs_to_cl_image,
                              DataType     data_type)
    {
        ARM_COMPUTE_UNUSED(export_rhs_to_cl_image);
        CLScheduler::get().default_reinit();

        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              context        = GpuWorkloadContext{&cl_compile_ctx};
        GpuWorkloadSketch sketch{&context};

        // Create sketch tensors
        ITensorInfo *lhs_info = context.create_tensor_info(TensorInfo(shape_a, 1, data_type));
        ITensorInfo *rhs_info = context.create_tensor_info(TensorInfo(shape_b, 1, data_type));
        ITensorInfo *dst_info = context.create_tensor_info();

        MatMulAttributes matmul_attr{};
        matmul_attr.adj_lhs(transpose_a);
        matmul_attr.adj_rhs(transpose_b);

        GpuMatMulSettings matmul_settings{};
        matmul_settings.m0(M0);
        matmul_settings.n0(N0);
        matmul_settings.k0(K0);

        ITensorInfo *ans_info = FunctionType::create_op(sketch, lhs_info, rhs_info, matmul_attr, matmul_settings);
        GpuOutput::create_op(sketch, ans_info, dst_info);

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        for (auto &data : runtime.get_auxiliary_tensors())
        {
            CLTensor     *tensor      = std::get<0>(data);
            TensorInfo    info        = std::get<1>(data);
            AuxMemoryInfo aux_mem_req = std::get<2>(data);
            tensor->allocator()->init(info, aux_mem_req.alignment);
            tensor->allocator()->allocate(); // Use ACL allocated memory
        }

        // Construct user tensors
        TensorType t_lhs{};
        TensorType t_rhs{};
        TensorType t_dst{};

        // Initialize user tensors
        t_lhs.allocator()->init(*lhs_info);
        t_rhs.allocator()->init(*rhs_info);
        t_dst.allocator()->init(*dst_info);

        ARM_COMPUTE_ASSERT(t_lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(t_rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(t_dst.info()->is_resizable());

        // Allocate and fill user tensors
        t_lhs.allocator()->allocate();
        t_rhs.allocator()->allocate();
        t_dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!t_lhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!t_rhs.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!t_dst.info()->is_resizable());

        fill(AccessorType(t_lhs), 0);
        fill(AccessorType(t_rhs), 1);

        // Run runtime
        runtime.run({&t_lhs, &t_rhs, &t_dst});

        return t_dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_a,
                                      const TensorShape &shape_b,
                                      const TensorShape &output_shape,
                                      bool               pretranspose_a,
                                      bool               pretranspose_b,
                                      DataType           data_type)
    {
        // We collapse dimensions > 3 onto dimension 3, i.e. 5D+ tensors will look like 4D
        // This is necessary unless we choose to extend gemm reference for 5D+ tensors
        TensorShape output_shape_collapsed = output_shape.collapsed_from(Window::DimZ);
        TensorShape shape_a_collapsed      = shape_a.collapsed_from(Window::DimZ);
        TensorShape shape_b_collapsed      = shape_b.collapsed_from(Window::DimZ);

        // Create reference
        SimpleTensor<T> a{shape_a_collapsed, data_type, 1};
        SimpleTensor<T> b{shape_b_collapsed, data_type, 1};
        SimpleTensor<T> c{output_shape_collapsed, data_type, 1};

        // Fill reference
        fill(a, 0);
        fill(b, 1);

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
        SimpleTensor<T> a_transposed{a_transposed_shape, data_type};
        SimpleTensor<T> b_transposed{b_transposed_shape, data_type};

        //pretranspose a if necessary
        if (pretranspose_a)
        {
            a_transposed = reference::permute<T>(a, PermutationVector(1U, 0U));
        }

        // pretranspose b if necessary
        if (pretranspose_b)
        {
            b_transposed = reference::permute<T>(b, PermutationVector(1U, 0U));
        }

        // Use transposed tensors if boolean enabled else use original tensors
        SimpleTensor<T> result =
            reference::gemm<T>((pretranspose_a) ? a_transposed : a, (pretranspose_b) ? b_transposed : b, c, 1.0f, 0.f);

        // We reshape the gemm output back if the tensor is high dimensional
        if (output_shape_collapsed != output_shape)
        {
            // std::cout << "called reshape: \n";
            result = reference::reshape_layer(result, output_shape);
        }

        return result;
    }

    CLTensor        _target{};
    SimpleTensor<T> _reference{};
    bool            _device_supports_export_to_cl_image{false};
    bool            _device_supports_mmul{false};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionGpuMatMulValidationFixture
    : public DynamicFusionGpuMatMulValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape lhs_shape,
               TensorShape rhs_shape,
               TensorShape output_shape,
               bool        transpose_a,
               bool        transpose_b,
               int         M0,
               int         N0,
               int         K0,
               bool        export_rhs_to_cl_image,
               DataType    data_type)
    {
        ARM_COMPUTE_UNUSED(export_rhs_to_cl_image);
        DynamicFusionGpuMatMulValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            lhs_shape, rhs_shape, output_shape, transpose_a, transpose_b, M0, N0, K0,
            false /* export_rhs_to_cl_image bias */, data_type);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_GPU_CL_MATMULKERNELFIXTURE_H
