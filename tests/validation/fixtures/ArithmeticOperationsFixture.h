/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ARITHMETIC_OPERATIONS_FIXTURE
#define ARM_COMPUTE_TEST_ARITHMETIC_OPERATIONS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticOperationGenericFixture : public framework::Fixture
{
public:
    void setup(reference::ArithmeticOperation op, const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, ActivationLayerInfo act_info, bool is_inplace)
    {
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

    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy,
                              QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        ref_src1  = create_tensor<TensorType>(shape0, data_type, 1, qinfo0);
        TensorType        ref_src2  = create_tensor<TensorType>(shape1, data_type, 1, qinfo1);
        TensorType        dst       = create_tensor<TensorType>(out_shape, data_type, 1, qinfo_out);

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if(_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) && (qinfo0 == qinfo_out);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) && (qinfo1 == qinfo_out);
            bool do_in_place     = out_shape.total_size() != 0 && (src1_is_inplace || src2_is_inplace);
            ARM_COMPUTE_ASSERT(do_in_place);

            if(src1_is_inplace)
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
        arith_op.configure(&ref_src1, &ref_src2, actual_dst, convert_policy, _act_info);

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!ref_src2.info()->is_resizable());

        // If don't do in-place computation, still need to allocate original dst
        if(!_is_inplace)
        {
            ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
            dst.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(ref_src1), 0);
        fill(AccessorType(ref_src2), 1);

        // Compute function
        arith_op.run();

        return std::move(*actual_dst);
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy,
                                      QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create reference
        SimpleTensor<T> ref_src1{ shape0, data_type, 1, qinfo0 };
        SimpleTensor<T> ref_src2{ shape1, data_type, 1, qinfo1 };
        SimpleTensor<T> ref_dst{ TensorShape::broadcast_shape(shape0, shape1), data_type, 1, qinfo_out };

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        auto result = reference::arithmetic_operation<T>(_op, ref_src1, ref_src2, ref_dst, convert_policy);
        return _act_info.enabled() ? reference::activation_layer(result, _act_info, qinfo_out) : result;
    }

    TensorType                     _target{};
    SimpleTensor<T>                _reference{};
    reference::ArithmeticOperation _op{ reference::ArithmeticOperation::ADD };
    ActivationLayerInfo            _act_info{};
    bool                           _is_inplace{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionBroadcastValidationFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape0, shape1, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionValidationFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionBroadcastValidationFloatFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape0, shape1, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionValidationFloatFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionValidationQuantizedFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy,
                                                                                            qinfo0, qinfo1, qinfo_out, ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticAdditionValidationQuantizedBroadcastFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out,
               bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::ADD, shape0, shape1, data_type, convert_policy,
                                                                                            qinfo0, qinfo1, qinfo_out, ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionBroadcastValidationFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape0, shape1, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionBroadcastValidationFloatFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, ActivationLayerInfo act_info,
               bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape0, shape1, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionValidationFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionValidationFloatFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy,
                                                                                            QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionValidationQuantizedFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy,
                                                                                            qinfo0, qinfo1, qinfo_out, ActivationLayerInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticSubtractionValidationQuantizedBroadcastFixture : public ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type, ConvertPolicy convert_policy, QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out,
               bool is_inplace)
    {
        ArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(reference::ArithmeticOperation::SUB, shape0, shape1, data_type, convert_policy,
                                                                                            qinfo0, qinfo1, qinfo_out, ActivationLayerInfo(), is_inplace);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ARITHMETIC_OPERATIONS_FIXTURE */
