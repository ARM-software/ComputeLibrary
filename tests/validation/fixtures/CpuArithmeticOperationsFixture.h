/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"

#if !defined(BARE_METAL)
#include <thread>
#include <vector>
#endif // !defined(BARE_METAL)

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr int NUM_THREADS = 3;
} // namespace
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
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
               bool                           is_inplace,
               TestType                       test_type)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _op                = op;
        _act_info          = act_info;
        _is_inplace        = is_inplace;
        _test_type         = test_type;
        _num_parallel_runs = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        compute_target(shape0, shape1, data_type, convert_policy, qinfo0, qinfo1, qinfo_out);
        compute_reference(shape0, shape1, data_type, convert_policy, qinfo0, qinfo1, qinfo_out);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    void allocate_and_fill_tensors(TensorType *src1, TensorType *src2, TensorType *dst)
    {
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            ARM_COMPUTE_ASSERT(src1[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(src2[i].info()->is_resizable());

            // Allocate tensors
            src1[i].allocator()->allocate();
            src2[i].allocator()->allocate();

            ARM_COMPUTE_ASSERT(!src1[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!src2[i].info()->is_resizable());

            // If don't do in-place computation, still need to allocate original dst
            if (!_is_inplace)
            {
                ARM_COMPUTE_ASSERT(dst[i].info()->is_resizable());
                dst[i].allocator()->allocate();
                ARM_COMPUTE_ASSERT(!dst[i].info()->is_resizable());
            }

            // Fill tensors
            fill(AccessorType(src1[i]), (2 * i + 0));
            fill(AccessorType(src2[i]), (2 * i + 1));
        }
    }

    void compute_target(const TensorShape &shape0,
                        const TensorShape &shape1,
                        DataType           data_type,
                        ConvertPolicy      convert_policy,
                        QuantizationInfo   qinfo0,
                        QuantizationInfo   qinfo1,
                        QuantizationInfo   qinfo_out)
    {
        // Create tensors
        TensorType src1[NUM_THREADS];
        TensorType src2[NUM_THREADS];
        TensorType dst[NUM_THREADS];

        ITensorPack run_pack[NUM_THREADS];

        TensorType *dst_ptrs[NUM_THREADS];

        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            src1[i]     = create_tensor<TensorType>(shape0, data_type, 1, qinfo0);
            src2[i]     = create_tensor<TensorType>(shape1, data_type, 1, qinfo1);
            dst[i]      = create_tensor<TensorType>(out_shape, data_type, 1, qinfo_out);
            dst_ptrs[i] = &dst[i];
        }

        // Check whether do in-place computation and whether inputs are broadcast compatible
        if (_is_inplace)
        {
            bool src1_is_inplace =
                !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) && (qinfo0 == qinfo_out);
            bool src2_is_inplace =
                !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) && (qinfo1 == qinfo_out);
            bool do_in_place = out_shape.total_size() != 0 && (src1_is_inplace || src2_is_inplace);
            ARM_COMPUTE_ASSERT(do_in_place);

            for (int i = 0; i < _num_parallel_runs; ++i)
            {
                dst_ptrs[i] = src1_is_inplace ? &(src1[i]) : &(src2[i]);
            }
        }

        // Create and configure function
        FunctionType arith_op;
        arith_op.configure(src1[0].info(), src2[0].info(), dst_ptrs[0]->info(), convert_policy, _act_info);

        allocate_and_fill_tensors(src1, src2, dst);

        if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL
            std::vector<std::thread> threads;

            threads.reserve(_num_parallel_runs);
            for (int i = 0; i < _num_parallel_runs; ++i)
            {
                // Compute function
                run_pack[i] = {{arm_compute::TensorType::ACL_SRC_0, &src1[i]},
                               {arm_compute::TensorType::ACL_SRC_1, &src2[i]},
                               {arm_compute::TensorType::ACL_DST, dst_ptrs[i]}};

                threads.emplace_back(
                    [&, i]
                    {
                        arith_op.run(run_pack[i]);
                        _target[i] = std::move(*(dst_ptrs[i]));
                    });
            }
            for (int i = 0; i < _num_parallel_runs; ++i)
            {
                threads[i].join();
            }
#endif // ifndef BARE_METAL
        }
        else
        {
            // Compute function
            ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &src1[0]},
                                 {arm_compute::TensorType::ACL_SRC_1, &src2[0]},
                                 {arm_compute::TensorType::ACL_DST, dst_ptrs[0]}};
            arith_op.run(run_pack);
            _target[0] = std::move(*(dst_ptrs[0]));
        }
    }

    void compute_reference(const TensorShape &shape0,
                           const TensorShape &shape1,
                           DataType           data_type,
                           ConvertPolicy      convert_policy,
                           QuantizationInfo   qinfo0,
                           QuantizationInfo   qinfo1,
                           QuantizationInfo   qinfo_out)
    {
        // Create reference
        SimpleTensor<T> src1{shape0, data_type, 1, qinfo0};
        SimpleTensor<T> src2{shape1, data_type, 1, qinfo1};
        SimpleTensor<T> ref_dst{TensorShape::broadcast_shape(shape0, shape1), data_type, 1, qinfo_out};

        // Fill reference
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            fill(src1, 2 * i + 0);
            fill(src2, 2 * i + 1);
            auto result   = reference::arithmetic_operation<T>(_op, src1, src2, ref_dst, convert_policy);
            _reference[i] = _act_info.enabled() ? reference::activation_layer(result, _act_info, qinfo_out) : result;
        }
    }

    TensorType                     _target[NUM_THREADS];
    SimpleTensor<T>                _reference[NUM_THREADS];
    reference::ArithmeticOperation _op{reference::ArithmeticOperation::ADD};
    ActivationLayerInfo            _act_info{};
    bool                           _is_inplace{};
    TestType                       _test_type{};
    int                            _num_parallel_runs{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticAdditionValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace, TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticSubtractionValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace, TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticAdditionThreadSafeValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace,
            TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticAdditionQuantizedThreadSafeValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape,
               DataType           data_type,
               ConvertPolicy      convert_policy,
               QuantizationInfo   qinfo0,
               QuantizationInfo   qinfo1,
               QuantizationInfo   qinfo_out,
               bool               is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::ADD, shape, shape, data_type, convert_policy, qinfo0, qinfo1, qinfo_out,
            ActivationLayerInfo(), is_inplace, TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticSubtractionThreadSafeValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, ConvertPolicy convert_policy, bool is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy, QuantizationInfo(),
            QuantizationInfo(), QuantizationInfo(), ActivationLayerInfo(), is_inplace,
            TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuArithmeticSubtractionQuantizedThreadSafeValidationFixture
    : public CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape,
               DataType           data_type,
               ConvertPolicy      convert_policy,
               QuantizationInfo   qinfo0,
               QuantizationInfo   qinfo1,
               QuantizationInfo   qinfo_out,
               bool               is_inplace)
    {
        CpuArithmeticOperationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            reference::ArithmeticOperation::SUB, shape, shape, data_type, convert_policy, qinfo0, qinfo1, qinfo_out,
            ActivationLayerInfo(), is_inplace, TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUARITHMETICOPERATIONSFIXTURE_H
