/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ELEMENTWISE_OPERATIONS_FIXTURE
#define ARM_COMPUTE_TEST_ELEMENTWISE_OPERATIONS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
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
class ArithmeticOperationsGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(ArithmeticOperation op, const TensorShape &shape0, const TensorShape &shape1,
               DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace = false, bool use_dynamic_shape = false)
    {
        _op                = op;
        _use_dynamic_shape = use_dynamic_shape;
        _is_inplace        = is_inplace;

        _target    = compute_target(shape0, shape1, data_type0, data_type1, output_data_type, qinfo0, qinfo1, qinfo_out);
        _reference = compute_reference(shape0, shape1, data_type0, data_type1, output_data_type, qinfo0, qinfo1, qinfo_out);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if(is_data_type_float(tensor.data_type()))
        {
            switch(_op)
            {
                case ArithmeticOperation::DIV:
                    library->fill_tensor_uniform_ranged(tensor, i, { std::pair<float, float>(-0.001f, 0.001f) });
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

    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
                              QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        ref_src1  = create_tensor<TensorType>(shape0, data_type0, 1, qinfo0);
        TensorType        ref_src2  = create_tensor<TensorType>(shape1, data_type1, 1, qinfo1);
        TensorType        dst       = create_tensor<TensorType>(out_shape, output_data_type, 1, qinfo_out);

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if(_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) && (qinfo0 == qinfo_out) && (data_type0 == output_data_type);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) && (qinfo1 == qinfo_out) && (data_type1 == output_data_type);
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

        // if _use_dynamic_shape is true, this fixture will test scenario for dynamic shapes.
        // - At configure time, all input tensors are marked as dynamic using set_tensor_dynamic()
        // - After configure, tensors are marked as static for run using set_tensor_static()
        // - The tensors with static shape are given to run()
        if(_use_dynamic_shape)
        {
            set_tensor_dynamic(ref_src1);
            set_tensor_dynamic(ref_src2);
        }

        // Create and configure function
        FunctionType elem_op;
        elem_op.configure(&ref_src1, &ref_src2, actual_dst);

        if(_use_dynamic_shape)
        {
            set_tensor_static(ref_src1);
            set_tensor_static(ref_src2);
        }

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();

        // If don't do in-place computation, still need to allocate original dst
        if(!_is_inplace)
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
        elem_op.run();

        return std::move(*actual_dst);
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1,
                                      DataType data_type0, DataType data_type1, DataType output_data_type,
                                      QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create reference
        SimpleTensor<T> ref_src1{ shape0, data_type0, 1, qinfo0 };
        SimpleTensor<T> ref_src2{ shape1, data_type1, 1, qinfo1 };
        SimpleTensor<T> ref_dst{ TensorShape::broadcast_shape(shape0, shape1), output_data_type, 1, qinfo_out };

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        return reference::arithmetic_operation<T>(_op, ref_src1, ref_src2, ref_dst);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    ArithmeticOperation _op{ ArithmeticOperation::ADD };
    bool                _use_dynamic_shape{ false };
    bool                _is_inplace{ false };
};

// Arithmetic operation fused with activation function
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticOperationsFuseActivationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(ArithmeticOperation op, const TensorShape &shape0, const TensorShape &shape1,
               DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, ActivationLayerInfo act_info, bool is_inplace = true)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(op, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
        _act_info   = act_info;
        _is_inplace = is_inplace;
    }

protected:
    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
                              QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        // Create tensors
        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        TensorType        ref_src1  = create_tensor<TensorType>(shape0, data_type0, 1, qinfo0);
        TensorType        ref_src2  = create_tensor<TensorType>(shape1, data_type1, 1, qinfo1);
        TensorType        dst       = create_tensor<TensorType>(out_shape, output_data_type, 1, qinfo_out);

        // Check whether do in-place computation and whether inputs are broadcast compatible
        TensorType *actual_dst = &dst;
        if(_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) && (qinfo0 == qinfo_out) && (data_type0 == output_data_type);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) && (qinfo1 == qinfo_out) && (data_type1 == output_data_type);
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
        FunctionType elem_op;
        elem_op.configure(&ref_src1, &ref_src2, actual_dst, _act_info);

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();

        // If don't do in-place computation, still need to allocate original dst
        if(!_is_inplace)
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
        elem_op.run();

        return std::move(*actual_dst);
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1,
                                      DataType data_type0, DataType data_type1, DataType output_data_type,
                                      QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)
    {
        auto result = ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::compute_reference(shape0, shape1, data_type0,
                                                                                                                       data_type1, output_data_type, qinfo0, qinfo1, qinfo_out);
        return _act_info.enabled() ? reference::activation_layer(result, _act_info, qinfo_out) : result;
    }

    ActivationLayerInfo _act_info{};
    bool                _is_inplace{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionBroadcastDynamicShapeValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace, true);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionDynamicShapeValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionBroadcastValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape0, shape1,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionValidationIntegerFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionValidationQuantizedFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::DIV, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxBroadcastValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape0, shape1,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxValidationQuantizedFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMaxQuantizedBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MAX, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinBroadcastValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape0, shape1,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinValidationQuantizedFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseMinQuantizedBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::MIN, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffBroadcastValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape0, shape1,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffValidationQuantizedFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwiseSquaredDiffQuantizedBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out, bool is_inplace)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::SQUARED_DIFF, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PReluLayerBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::PRELU, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PReluLayerValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::PRELU, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PReluLayerValidationQuantizedFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::PRELU, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PReluLayerQuantizedBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type,
               QuantizationInfo qinfo0, QuantizationInfo qinfo1, QuantizationInfo qinfo_out)

    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::PRELU, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             qinfo0, qinfo1, qinfo_out);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwisePowerBroadcastValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::POWER, shape0, shape1,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwisePowerValidationFixture : public ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, bool is_inplace)
    {
        ArithmeticOperationsGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::POWER, shape, shape,
                                                                                             data_type0, data_type1, output_data_type,
                                                                                             QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwisePowerBroadcastValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::POWER, shape0, shape1,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ElementwisePowerValidationFloatFixture : public ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type, ActivationLayerInfo act_info, bool is_inplace)
    {
        ArithmeticOperationsFuseActivationFixture<TensorType, AccessorType, FunctionType, T>::setup(ArithmeticOperation::POWER, shape, shape,
                                                                                                    data_type0, data_type1, output_data_type,
                                                                                                    QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), act_info, is_inplace);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ARITHMETIC_OPERATIONS_FIXTURE */
