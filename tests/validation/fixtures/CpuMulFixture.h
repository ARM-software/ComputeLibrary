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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

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
constexpr int NUM_THREADS =  3;
}// namespace
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3>
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
               bool               is_inplace,
               QuantizationInfo    qinfo1,
               QuantizationInfo    qinfo2,
               QuantizationInfo    qinfo_out,
               TestType            test_type)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            (dt_in1 == DataType::F16 || dt_in2 == DataType::F16 || dt_out == DataType::F16) &&
            !CPUInfo::get().has_fp16())
        {
            return;
        }

        _is_inplace = is_inplace;
        _test_type = test_type;

        _num_parallel_runs        = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        compute_target(shape0, shape1, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy,
                                     qinfo1, qinfo2, qinfo_out, ActivationLayerInfo());

        compute_reference(shape0, shape1, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy,
                              qinfo1, qinfo2, qinfo_out, ActivationLayerInfo());
    }

protected:
    template <typename U>
    void fill(U &&tensor, unsigned int seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    void allocate_and_fill_tensors(TensorType *src1, TensorType *src2, TensorType *dst){
        auto allocate_tensor = [](TensorType &t)
        {
            ARM_COMPUTE_ASSERT(t.info()->is_resizable());
            t.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!t.info()->is_resizable());
        };

        for(int i = 0; i < _num_parallel_runs; ++i){
            allocate_tensor(src1[i]);
            allocate_tensor(src2[i]);

            // If don't do in-place computation, still need to allocate original dst
            if (!_is_inplace)
            {
                allocate_tensor(dst[i]);
            }

            // Fill tensors
            fill(AccessorType(src1[i]), (2*i + 0));
            fill(AccessorType(src2[i]), (2*i + 1));
        }
    }

    void compute_target(const TensorShape  &shape0,
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
        TensorType src1[NUM_THREADS];
        TensorType src2[NUM_THREADS];
        TensorType dst[NUM_THREADS];

        TensorType *dst_ptrs[NUM_THREADS];

        ITensorPack run_pack[NUM_THREADS];

        const TensorShape out_shape = TensorShape::broadcast_shape(shape0, shape1);

        for(int i = 0; i < _num_parallel_runs; ++i){
            src1[i] = create_tensor<TensorType>(shape0, dt_in1, 1, qinfo0);
            src2[i] = create_tensor<TensorType>(shape1, dt_in2, 1, qinfo1);
            dst[i]  = create_tensor<TensorType>(out_shape, dt_out, 1, qinfo_out);
            dst_ptrs[i] = &dst[i];

        }

        // Check whether do in-place computation and whether inputs are broadcast compatible
        if (_is_inplace)
        {
            bool src1_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape0, 0) &&
                                   (qinfo0 == qinfo_out) && (dt_in1 == dt_out);
            bool src2_is_inplace = !arm_compute::detail::have_different_dimensions(out_shape, shape1, 0) &&
                                   (qinfo1 == qinfo_out) && (dt_in2 == dt_out);
            bool do_in_place = out_shape.total_size() != 0 && (src1_is_inplace || src2_is_inplace);
            ARM_COMPUTE_ASSERT(do_in_place);

            for(int i = 0; i < _num_parallel_runs; ++i){
                dst_ptrs[i] = src1_is_inplace ? &(src1[i]) : &(src2[i]);
            }
        }

        // Create and configure function
        FunctionType multiply;
        multiply.configure(src1[0].info(), src2[0].info(), dst_ptrs[0]->info(), scale, convert_policy, rounding_policy,
                           act_info);

        allocate_and_fill_tensors(src1, src2, dst);

         if(_test_type == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL
            std::vector<std::thread> threads;

            threads.reserve(_num_parallel_runs);
            for(int i = 0; i < _num_parallel_runs; ++i)
            {
                // Compute function
                run_pack[i] = { { arm_compute::TensorType::ACL_SRC_0, &src1[i] },
                {arm_compute::TensorType::ACL_SRC_1, &src2[i]},
                {arm_compute::TensorType::ACL_DST, dst_ptrs[i]}};

                threads.emplace_back([&,i]
                {

                    multiply.run(run_pack[i]);
                    _target[i] =std::move(*(dst_ptrs[i]));
                });
            }
            for(int i = 0; i < _num_parallel_runs; ++i)
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
            multiply.run(run_pack);

            _target[0] = std::move(*(dst_ptrs[0]));
        }
    }

    void compute_reference(const TensorShape  &shape0,
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

        for(int i = 0; i < _num_parallel_runs; ++i)
        {
            // Fill reference
            fill(src1, 2*i + 0);
            fill(src2, 2*i + 1);
            auto result = reference::pixel_wise_multiplication<T1, T2, T3>(src1, src2, scale, convert_policy,
                                                                        rounding_policy, dt_out, qinfo_out);
            _reference[i] = act_info.enabled() ? reference::activation_layer(result, act_info, qinfo_out) : result;
        }
    }

    TensorType                     _target[NUM_THREADS];
    SimpleTensor<T3>               _reference[NUM_THREADS];
    bool                           _is_inplace{false};
    TestType                       _test_type{};
    int                            _num_parallel_runs{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3>
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
            shape, shape, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy, is_inplace, QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3>
class CpuMulThreadSafeValidationFixture
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
            shape, shape, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy, is_inplace, QuantizationInfo(), QuantizationInfo(), QuantizationInfo(), TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2, typename T3>
class CpuMulQuantizedThreadSafeValidationFixture
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
               QuantizationInfo qinfo1,
               QuantizationInfo qinfo2,
               QuantizationInfo    qinfo_out,
               bool               is_inplace)
    {
        CpuMulGenericValidationFixture<TensorType, AccessorType, FunctionType, T1, T2, T3>::setup(
            shape, shape, dt_in1, dt_in2, dt_out, scale, convert_policy, rounding_policy, is_inplace, qinfo1, qinfo2, qinfo_out, TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUMULFIXTURE_H
