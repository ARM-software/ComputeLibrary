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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/SoftmaxLayer.h"
#include <random>

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
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool IS_LOG = false>
class CpuSoftmaxValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis, QuantizationInfo qinfo,
               TestType test_type = TestType::ConfigureOnceRunOnce)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        quantization_info_  = qinfo;
        test_type_          = test_type;
        num_parallel_runs_  = (test_type_ == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        compute_reference(shape, data_type, quantization_info_, beta, axis);
        compute_target(shape, data_type, quantization_info_, beta, axis);
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
        for(int i = 0; i < num_parallel_runs_; ++i){

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

    void compute_target(const TensorShape &shape, DataType data_type,
                              QuantizationInfo quantization_info, float beta, int32_t axis)
    {
        TensorType src[NUM_THREADS];
        TensorType dst[NUM_THREADS];
        ITensorPack run_pack[NUM_THREADS];
        TensorType *dst_ptrs[NUM_THREADS];
        auto mg = MemoryGroup{};

        // Create tensors
        for(int i = 0; i < num_parallel_runs_; ++i){
            src[i]  = create_tensor<TensorType>(shape, data_type, 1, quantization_info);
            dst[i]  = create_tensor<TensorType>(shape, data_type, 1, get_softmax_output_quantization_info(data_type, IS_LOG));
            dst_ptrs[i] = &dst[i];
        }

        // Create and configure function
        FunctionType softmax;
        softmax.configure(src[0].info(), dst[0].info(), beta, axis);

        allocate_and_fill_tensors(src, dst);

        if(test_type_ == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL
            std::vector<std::thread> threads;

            threads.reserve(num_parallel_runs_);
            for(int i = 0; i < num_parallel_runs_; ++i)
            {
                // Compute function
                run_pack[i] = {{arm_compute::TensorType::ACL_SRC_0, &src[i]},
                               {arm_compute::TensorType::ACL_DST, dst_ptrs[i]}};

                threads.emplace_back([&,i]
                {
                    auto ws = manage_workspace<Tensor>(softmax.workspace(), mg, run_pack[i]);
                    softmax.run(run_pack[i]);
                    target_[i] = std::move(*(dst_ptrs[i]));
                });
            }
            for(int i = 0; i < num_parallel_runs_; ++i)
            {
                threads[i].join();
            }
#endif // ifndef BARE_METAL
        }
        else
        {
            // Compute function
            ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &src[0]},
                                 {arm_compute::TensorType::ACL_DST, dst_ptrs[0]}};
            auto ws = manage_workspace<Tensor>(softmax.workspace(), mg, run_pack);

            // Compute function
            softmax.run(run_pack);
            target_[0] = std::move(*(dst_ptrs[0]));
        }
    }

    void compute_reference(const TensorShape &shape, DataType data_type,
                                      QuantizationInfo quantization_info, float beta, int32_t axis)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, quantization_info };

        // Fill reference
        for(int i = 0; i < num_parallel_runs_; ++i)
        {
            // Fill reference
            fill(src);
            reference_[i] = reference::softmax_layer<T>(src, beta, axis, IS_LOG);
        }
    }

    TensorType       target_[NUM_THREADS];
    SimpleTensor<T>  reference_[NUM_THREADS];
    QuantizationInfo quantization_info_{};
    TestType         test_type_{};
    int              num_parallel_runs_{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuSoftmaxValidationFixture
    : public CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis)
    {
        CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, data_type, beta, axis, QuantizationInfo(), TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuSoftmaxThreadSafeValidationFixture
    : public CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis)
    {
        CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, data_type, beta, axis, QuantizationInfo(), TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuSoftmaxQuantizedThreadSafeValidationFixture
    : public CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, DataType data_type, float beta, size_t axis, QuantizationInfo qinfo)
    {
        CpuSoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, data_type, beta, axis, qinfo, TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUSOFTMAXFIXTURE_H
