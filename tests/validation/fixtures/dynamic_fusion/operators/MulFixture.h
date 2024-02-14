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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_MULFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_MULFIXTURE_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
/* We use a separate test fixture for Multiplication op instead of reusing ElementwiseBinaryFixture to avoid exposing
 * the internal enum ElementwiseOp to the public utils/TypePrinters.h as required by the data test case macros
 * to print the test data.
 */
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionMulValidationFixture : public framework::Fixture
{
public:
    void setup(const TensorShape &shape0,
               const TensorShape &shape1,
               const TensorShape &shape2,
               DataType           data_type,
               bool               is_inplace,
               bool               fuse_two_ops = false)
    {
        _data_type  = data_type;
        _is_inplace = is_inplace;
        _fuse       = fuse_two_ops;
        ARM_COMPUTE_ERROR_ON_MSG(_fuse && shape2.total_size() == 0, "No shape2 provided for fusion of two ops.");
        ARM_COMPUTE_ERROR_ON_MSG(_fuse && _is_inplace, "In place for fusing case not supported yet.");
        _target    = compute_target(shape0, shape1, shape2);
        _reference = compute_reference(shape0, shape1, shape2);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, const TensorShape &shape2)
    {
        // Create a new workload sketch
        auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
        auto              context        = GpuWorkloadContext{&cl_compile_ctx};
        GpuWorkloadSketch sketch{&context};

        // Fuse first multiplication op
        ITensorInfo *lhs_info = context.create_tensor_info(TensorInfo(shape0, 1, _data_type));
        ITensorInfo *rhs_info = context.create_tensor_info(TensorInfo(shape1, 1, _data_type));
        ITensorInfo *dst_info = context.create_tensor_info();

        ITensorInfo *rhs_info_fuse = nullptr;

        ITensorInfo *ans_info = FunctionType::create_op(sketch, lhs_info, rhs_info);

        if (_fuse)
        {
            rhs_info_fuse          = context.create_tensor_info(TensorInfo(shape2, 1, _data_type));
            ITensorInfo *ans2_info = FunctionType::create_op(sketch, ans_info, rhs_info_fuse);
            GpuOutput::create_op(sketch, ans2_info, dst_info);
        }
        else
        {
            GpuOutput::create_op(sketch, ans_info, dst_info);
        }

        // Configure runtime
        ClWorkloadRuntime runtime;
        runtime.configure(sketch);

        // (Important) Allocate auxiliary tensor memory if there are any
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
        TensorType t_rhs_fuse{};
        TensorType t_dst{};

        // Initialize user tensors
        t_lhs.allocator()->init(*lhs_info);
        t_rhs.allocator()->init(*rhs_info);
        t_dst.allocator()->init(*dst_info);
        if (_fuse)
        {
            t_rhs_fuse.allocator()->init(*rhs_info_fuse);
        }

        // Allocate and fill user tensors
        // Instead of using ACL allocator, the user can choose to import memory into the tensors
        t_lhs.allocator()->allocate();
        t_rhs.allocator()->allocate();
        t_dst.allocator()->allocate();
        if (_fuse)
        {
            t_rhs_fuse.allocator()->allocate();
        }

        fill(AccessorType(t_lhs), 0);
        fill(AccessorType(t_rhs), 1);
        if (_fuse)
        {
            fill(AccessorType(t_rhs_fuse), 2);
        }

        // Run runtime
        if (_fuse)
        {
            runtime.run({&t_lhs, &t_rhs, &t_rhs_fuse, &t_dst});
        }
        else
        {
            runtime.run({&t_lhs, &t_rhs, &t_dst});
        }

        return t_dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, const TensorShape &shape2)
    {
        // Create reference
        SimpleTensor<T> ref_lhs{shape0, _data_type, 1, QuantizationInfo()};
        SimpleTensor<T> ref_rhs{shape1, _data_type, 1, QuantizationInfo()};
        SimpleTensor<T> ref_rhs_fuse{shape2, _data_type, 1, QuantizationInfo()};

        // Fill reference
        fill(ref_lhs, 0);
        fill(ref_rhs, 1);
        SimpleTensor<T> ref_dst = reference::pixel_wise_multiplication<T, T, T>(
            ref_lhs, ref_rhs, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_UP, _data_type,
            QuantizationInfo());
        if (_fuse)
        {
            fill(ref_rhs_fuse, 2);
            SimpleTensor<T> ref_dst_fuse = reference::pixel_wise_multiplication<T, T, T>(
                ref_dst, ref_rhs_fuse, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_UP, _data_type,
                QuantizationInfo());
            return ref_dst_fuse;
        }
        return ref_dst;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
    bool            _is_inplace{false};
    bool            _fuse{false};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionMulOneOpValidationFixture
    : public DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, DataType data_type, bool is_inplace)
    {
        DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape0, shape0, TensorShape(), data_type, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionMulBroadcastValidationFixture
    : public DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, bool is_inplace)
    {
        DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape0, shape1, TensorShape(), data_type, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DynamicFusionMulTwoOpsValidationFixture
    : public DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0,
               const TensorShape &shape1,
               const TensorShape &shape2,
               DataType           data_type,
               bool               is_inplace,
               bool               fuse_two_ops)
    {
        DynamicFusionMulValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape0, shape1, shape2, data_type, is_inplace, fuse_two_ops);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_DYNAMIC_FUSION_OPERATORS_MULFIXTURE_H
