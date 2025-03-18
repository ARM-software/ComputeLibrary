/*
 * Copyright (c) 2018-2021, 2023-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUELEMENTWISEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUELEMENTWISEFIXTURE_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuElementwiseOperationsGenericFixture : public framework::Fixture
{
public:
    void setup(ArithmeticOperation op,
               const TensorShape  &shape0,
               const TensorShape  &shape1,
               DataType            data_type0,
               DataType            data_type1,
               DataType            output_data_type,
               bool                is_inplace = false)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            (data_type0 == DataType::F16 || data_type1 == DataType::F16 || output_data_type == DataType::F16) &&
            !CPUInfo::get().has_fp16())
        {
            return;
        }

        _op         = op;
        _is_inplace = is_inplace;

        _target    = compute_target(shape0, shape1, data_type0, data_type1, output_data_type);
        _reference = compute_reference(shape0, shape1, data_type0, data_type1, output_data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if (is_data_type_float(tensor.data_type()))
        {
            switch (_op)
            {
                case ArithmeticOperation::DIV:
                    library->fill_tensor_uniform_ranged(tensor, i, {std::pair<float, float>(-0.001f, 0.001f)});
                    break;
                case ArithmeticOperation::POWER:
                    library->fill_tensor_uniform(tensor, i, 0.0f, 5.0f);
                    break;
                default:
                    library->fill_tensor_uniform(tensor, i);
            }
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &shape0,
                              const TensorShape &shape1,
                              DataType           data_type0,
                              DataType           data_type1,
                              DataType           output_data_type)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        ref_src1  = create_tensor<TensorType>(shape0, data_type0, 1, QuantizationInfo());
        TensorType        ref_src2  = create_tensor<TensorType>(shape1, data_type1, 1, QuantizationInfo());
        TensorType        dst       = create_tensor<TensorType>(out_shape, output_data_type, 1, QuantizationInfo());

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if (_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) &&
                                   (data_type0 == output_data_type);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) &&
                                   (data_type1 == output_data_type);
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
        FunctionType elem_op;
        elem_op.configure(ref_src1.info(), ref_src2.info(), actual_dst->info());

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();

        // If don't do in-place computation, still need to allocate original dst
        if (!_is_inplace)
        {
            ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        ARM_COMPUTE_ASSERT(!ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!ref_src2.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(ref_src1), 0);
        fill(AccessorType(ref_src2), 1);

        // Compute function
        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &ref_src1},
                             {arm_compute::TensorType::ACL_SRC_1, &ref_src2},
                             {arm_compute::TensorType::ACL_DST, actual_dst}

        };

        elem_op.run(run_pack);

        return std::move(*actual_dst);
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0,
                                      const TensorShape &shape1,
                                      DataType           data_type0,
                                      DataType           data_type1,
                                      DataType           output_data_type)
    {
        // Create reference
        SimpleTensor<T> ref_src1{shape0, data_type0, 1, QuantizationInfo()};
        SimpleTensor<T> ref_src2{shape1, data_type1, 1, QuantizationInfo()};
        SimpleTensor<T> ref_dst{TensorShape::broadcast_shape(shape0, shape1), output_data_type, 1, QuantizationInfo()};

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        return reference::arithmetic_operation<T>(_op, ref_src1, ref_src2, ref_dst);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    ArithmeticOperation _op{ArithmeticOperation::ADD};
    bool                _is_inplace{false};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuElementwiseDivisionValidationFixture
    : public CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(
        const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            ArithmeticOperation::DIV, shape, shape, data_type0, data_type1, output_data_type, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuElementwiseMaxValidationFixture
    : public CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(
        const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            ArithmeticOperation::MAX, shape, shape, data_type0, data_type1, output_data_type, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuElementwiseMinValidationFixture
    : public CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(
        const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            ArithmeticOperation::MIN, shape, shape, data_type0, data_type1, output_data_type, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuPReluValidationFixture
    : public CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(
        const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        CpuElementwiseOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            ArithmeticOperation::PRELU, shape, shape, data_type0, data_type1, output_data_type, is_inplace);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUELEMENTWISEFIXTURE_H
