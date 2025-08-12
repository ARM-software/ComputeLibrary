/*
 * Copyright (c) 2024-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUTRANSPOSEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUTRANSPOSEFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Permute.h"

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
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuTransposeGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType data_type, QuantizationInfo qinfo, TestType test_type = TestType::ConfigureOnceRunOnce)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }
        _test_type  = test_type;
        _num_parallel_runs = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        compute_target(shape, data_type, qinfo);
        compute_reference(shape, data_type, qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
            library->fill(tensor, distribution, 0);
        }
        else if(tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -10.0f, 10.0f };
            library->fill(tensor, distribution, 0);
        }
        else if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    void allocate_and_fill_tensors(TensorType *src, TensorType *dst){
        for(int i = 0; i < _num_parallel_runs; ++i) {

            ARM_COMPUTE_ASSERT(src[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(dst[i].info()->is_resizable());

            // Allocate tensors
            src[i].allocator()->allocate();
            dst[i].allocator()->allocate();

            ARM_COMPUTE_ASSERT(!src[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!dst[i].info()->is_resizable());

            // Fill tensors
            fill(AccessorType(src[i]));
        }
    }

    void compute_target(const TensorShape &shape, DataType data_type, QuantizationInfo qinfo)
    {
        // Create tensors
        TensorType src[NUM_THREADS];
        TensorType dst[NUM_THREADS];
        TensorType *dst_ptrs[NUM_THREADS];

        // Retain the shape but make rows the columns of the original shape
        TensorShape output_shape = shape;
        std::swap(output_shape[0], output_shape[1]);

        for(int i = 0; i < _num_parallel_runs; ++i){
            src[i] = create_tensor<TensorType>(shape, data_type, 1, qinfo);
            dst[i] = create_tensor<TensorType>(output_shape, data_type, 1, qinfo);
            dst_ptrs[i] = &dst[i];
        }

        // Create and configure function
        FunctionType trans_func;
        trans_func.configure(src[0].info(), dst_ptrs[0]->info());

        allocate_and_fill_tensors(src, dst);

        if(_test_type == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL

            ITensorPack run_pack[NUM_THREADS];
            std::vector<std::thread> threads;

            threads.reserve(_num_parallel_runs);
            for(int i = 0; i < _num_parallel_runs; ++i)
            {
                // Compute function
                run_pack[i] = { {arm_compute::TensorType::ACL_SRC, &src[i]},
                                {arm_compute::TensorType::ACL_DST, dst_ptrs[i]}};

                threads.emplace_back([&,i]
                {
                    trans_func.run(run_pack[i]);
                    _target[i] = std::move(*(dst_ptrs[i]));
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
            ITensorPack run_pack{{ arm_compute::TensorType::ACL_SRC, &src[0]},
                {arm_compute::TensorType::ACL_DST, dst_ptrs[0]}};
            trans_func.run(run_pack);
            _target[0] = std::move(*(dst_ptrs[0]));
        }
    }

    void compute_reference(const TensorShape &shape, DataType data_type, QuantizationInfo qinfo)
    {
        // Create reference
        SimpleTensor<T> src{shape, data_type, 1, qinfo};

        for(int i = 0; i < _num_parallel_runs; ++i)
        {
            // Fill reference
            fill(src);
            _reference[i] = reference::permute<T>(src, PermutationVector(1U, 0U));
        }
    }

    TensorType      _target[NUM_THREADS];
    SimpleTensor<T> _reference[NUM_THREADS];
    TestType        _test_type{};
    int             _num_parallel_runs{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuTransposeValidationFixture
    : public CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuTransposeThreadSafeValidationFixture
    : public CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type)
    {
        CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, QuantizationInfo(),
            TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuTransposeQuantizedThreadSafeValidationFixture
    : public CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(const TensorShape &shape, DataType data_type, QuantizationInfo qinfo)
    {
        CpuTransposeGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, qinfo,
            TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUTRANSPOSEFIXTURE_H
