/*
 * Copyright (c) 2017-2021, 2023-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3 = T2>
class CpuMulGenericValidationFixture : public framework::Fixture
{
public:
    void setup(const TensorShape &shape0,
               const TensorShape &shape1,
               DataType           dt_in1,
               DataType           dt_in2,
               DataType           dt_out,
               float              scale,
               ConvertPolicy      convert_policy,
               RoundingPolicy     rounding_policy,
               bool               is_inplace)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            (dt_in1 == DataType::F16 || dt_in2 == DataType::F16 || dt_out == DataType::F16) &&
            !CPUInfo::get().has_fp16())
        {
            return;
        }

        _is_inplace = is_inplace;
        _target     = compute_target(shape0, shape1, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy,
                                     QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo());
        _reference =
            compute_reference(shape0, shape1, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy,
                              QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo());
    }

protected:
    template <typename U>
    void fill(U &&tensor, unsigned int seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    TensorType compute_target(const TensorShape  &shape0,
                              const TensorShape  &shape1,
                              DataType            dt_in1,
                              DataType            dt_in2,
                              DataType            dt_out,
                              float               scale,
                              ConvertPolicy       convert_policy,
                              RoundingPolicy      rounding_policy,
                              QuantizationInfo    qinfo0,
                              QuantizationInfo    qinfo1,
                              QuantizationInfo    qinfo_out,
                              ActivationLayerInfo act_info)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        src1      = create_tensor<TensorType>(shape0, dt_in1, 1, qinfo0);
        TensorType        src2      = create_tensor<TensorType>(shape1, dt_in2, 1, qinfo1);
        TensorType        dst       = create_tensor<TensorType>(out_shape, dt_out, 1, qinfo_out);

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if (_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) &&
                                   (qinfo0 == qinfo_out) && (dt_in1 == dt_out);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) &&
                                   (qinfo1 == qinfo_out) && (dt_in2 == dt_out);
            bool do_in_place = out_shape.total_size() != 0 && (src1_is_inplace || src2_is_inplace);
            ARM_COMPUTE_ASSERT(do_in_place);

            if (src1_is_inplace)
            {
                actual_dst = &src1;
            }
            else
            {
                actual_dst = &src2;
            }
        }

        auto allocate_tensor = [](TensorType &t)
        {
            ARM_COMPUTE_ASSERT(t.info()->is_resizable());
            t.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!t.info()->is_resizable());
        };

        // Create and configure function
        FunctionType multiply;
        multiply.configure(src1.info(), src2.info(), actual_dst->info(), scale, convert_policy, rounding_policy,
                           act_info);

        allocate_tensor(src1);
        allocate_tensor(src2);

        // If don't do in-place computation, still need to allocate original dst
        if (!_is_inplace)
        {
            allocate_tensor(dst);
        }

        // Fill tensors
        fill(AccessorType(src1), 0);
        fill(AccessorType(src2), 1);

        // Compute function
        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &src1},
                             {arm_compute::TensorType::ACL_SRC_1, &src2},
                             {arm_compute::TensorType::ACL_DST, actual_dst}};
        multiply.run(run_pack);

        return std::move(*actual_dst);
    }

    SimpleTensor<T3> compute_reference(const TensorShape  &shape0,
                                       const TensorShape  &shape1,
                                       DataType            dt_in1,
                                       DataType            dt_in2,
                                       DataType            dt_out,
                                       float               scale,
                                       ConvertPolicy       convert_policy,
                                       RoundingPolicy      rounding_policy,
                                       QuantizationInfo    qinfo0,
                                       QuantizationInfo    qinfo1,
                                       QuantizationInfo    qinfo_out,
                                       ActivationLayerInfo act_info)
    {
        // Create reference
        SimpleTensor<T1> src1{shape0, dt_in1, 1, qinfo0};
        SimpleTensor<T2> src2{shape1, dt_in2, 1, qinfo1};

        // Fill reference
        fill(src1, 0);
        fill(src2, 1);

        auto result = reference::pixel_wise_multiplication<T1, T2, T3>(src1, src2, scale, convert_policy,
                                                                       rounding_policy, dt_out, qinfo_out);
        return act_info.enabled() ? reference::activation_layer(result, act_info, qinfo_out) : result;
    }

    TensorType       _target{};
    SimpleTensor<T3> _reference{};
    bool             _is_inplace{false};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3 = T2>
class CpuMulValidationFixture
    : public CpuMulGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2, T3>
{
public:
    void setup(const TensorShape &shape,
               DataType           dt_in1,
               DataType           dt_in2,
               DataType           dt_out,
               float              scale,
               ConvertPolicy      convert_policy,
               RoundingPolicy     rounding_policy,
               bool               is_inplace)
    {
        CpuMulGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2, T3>::setup(
            shape, shape, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy, is_inplace);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H
