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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUARITHMETICOPERATIONSFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUARITHMETICOPERATIONSFIXTURE_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorShape.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"

#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuArithmeticOperationGenericFixture : public framework::Fixture
{
public:
    void setup(reference::ArithmeticOperation op,
               const TensorShape             &shape0,
               const TensorShape             &shape1,
               DataType                       data_type,
               ConvertPolicy                  convert_policy,
               QuantizationInfo               qinfo0,
               QuantizationInfo               qinfo1,
               QuantizationInfo               qinfo_out,
               ActivationLayerInfo            act_info,
               bool                           is_inplace)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _op         = op;
        _act_info   = act_info;
        _is_inplace = is_inplace;
        _target     = compute_target(shape0, shape1, data_type, convert_policy, qinfo0, qinfo1, qinfo_out);
        _reference  = compute_reference(shape0, shape1, data_type, convert_policy, qinfo0, qinfo1, qinfo_out);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape0,
                              const TensorShape &shape1,
                              DataType           data_type,
                              ConvertPolicy      convert_policy,
                              QuantizationInfo   qinfo0,
                              QuantizationInfo   qinfo1,
                              QuantizationInfo   qinfo_out)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        ref_src1  = create_tensor<TensorType>(shape0, data_type, 1, qinfo0);
        TensorType        ref_src2  = create_tensor<TensorType>(shape1, data_type, 1, qinfo1);
        TensorType        dst       = create_tensor<TensorType>(out_shape, data_type, 1, qinfo_out);

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if (_is_inplace)
        {
            bool src1_is_inplace =
                !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) && (qinfo0 == qinfo_out);
            bool src2_is_inplace =
                !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) && (qinfo1 == qinfo_out);
            bool do_in_place = out_shape.total_size() != 0 && (src1_is_inplace || src2_is_inplace);
            ARM_COMPUTE_ASSERT(do_in_place);

            if (src1_is_inplace)
            {
                actual_dst = &ref_src1;
            }
            else
            {
                actual_dst = &ref_src2;
            }
        }

        // Create and configure function
        FunctionType arith_op;
        arith_op.configure(ref_src1.info(), ref_src2.info(), actual_dst->info(), convert_policy, _act_info);

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!ref_src2.info()->is_resizable());

        // If don't do in-place computation, still need to allocate original dst
        if (!_is_inplace)
        {
            ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(ref_src1), 0);
        fill(AccessorType(ref_src2), 1);

        // Compute function
        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &ref_src1},
                             {arm_compute::TensorType::ACL_SRC_1, &ref_src2},
                             {arm_compute::TensorType::ACL_DST, &dst}};
        arith_op.run(run_pack);

        return std::move(*actual_dst);
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &shape0,
                                            const TensorShape &shape1,
                                            DataType           data_type,
                                            ConvertPolicy      convert_policy,
                                            QuantizationInfo   qinfo0,
                                            QuantizationInfo   qinfo1,
                                            QuantizationInfo   qinfo_out)
    {
        // Create reference
        SimpleTensor<uint8_t> ref_src1{shape0, data_type, 1, qinfo0};
        SimpleTensor<uint8_t> ref_src2{shape1, data_type, 1, qinfo1};
        SimpleTensor<uint8_t> ref_dst{TensorShape::broadcast_shape(shape0, shape1), data_type, 1, qinfo_out};

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        auto result = reference::arithmetic_operation<uint8_t>(_op, ref_src1, ref_src2, ref_dst, convert_policy);
        return _act_info.enabled() ? reference::activation_layer(result, _act_info, qinfo_out) : result;
    }

    TensorType                     _target{};
    SimpleTensor<uint8_t>          _reference{};
    reference::ArithmeticOperation _op{reference::ArithmeticOperation::ADD};
    ActivationLayerInfo            _act_info{};
    bool                           _is_inplace{};
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuArithmeticAdditionValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType>::setup(
            reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuArithmeticSubtractionValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType>::setup(
            reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUARITHMETICOPERATIONSFIXTURE_H
