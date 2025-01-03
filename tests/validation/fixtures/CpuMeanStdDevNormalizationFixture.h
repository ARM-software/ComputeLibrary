/*
 * Copyright (c) 2019-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUMEANSTDDEVNORMALIZATIONFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUMEANSTDDEVNORMALIZATIONFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/MeanStdDevNormalizationLayer.h"

#ifndef BARE_METAL
#include <thread>
#include <vector>
#endif // ifndef BARE_METAL

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
class CpuMeanStdDevNormalizationValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape      shape,
               bool             in_place,
               float            epsilon,
               DataType         data_type,
               QuantizationInfo quantization_info,
               TestType         test_type)
    {
        if (std::is_same<TensorType, Tensor>::value && data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _in_place                 = in_place;
        _data_type                = data_type;
        _test_type                = test_type;
        _num_parallel_runs  = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);
        _output_quantization_info = QuantizationInfo(0.025f, 110);
        _input_quantization_info  = in_place ? _output_quantization_info : quantization_info;

        compute_target(shape, epsilon);
        compute_reference(shape, epsilon);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int seed_offset)
    {
        if(is_data_type_float(_data_type))
        {
            std::uniform_real_distribution<> distribution{ -1.0f, 1.0f };
            if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
            {
                library->fill(tensor, distribution, seed_offset);
            }
            else
            {
                library->fill(tensor, distribution, 0);
            }
        }
        else
        {
            std::uniform_int_distribution<> distribution{ 0, 255 };
            library->fill(tensor, distribution, 0);
        }
    }

    void allocate_and_fill_tensors(TensorType *src, TensorType *dst)
    {
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            ARM_COMPUTE_ASSERT(src[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(dst[i].info()->is_resizable());

            // Allocate tensors
            src[i].allocator()->allocate();
            ARM_COMPUTE_ASSERT(!src[i].info()->is_resizable());

            if (!_in_place)
            {
                dst[i].allocator()->allocate();
                ARM_COMPUTE_ASSERT(!dst[i].info()->is_resizable());
            }

            // Fill tensors
            fill(AccessorType(src[i]), i);
        }
    }

    void compute_target(TensorShape shape, float epsilon)
    {
        // Create tensors
        TensorType  src[NUM_THREADS];
        TensorType  dst[NUM_THREADS];
        ITensorPack run_pack[NUM_THREADS];

        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            src[i] = create_tensor<TensorType>(shape, _data_type, 1, _input_quantization_info);
            dst[i] = create_tensor<TensorType>(shape, _data_type, 1, _output_quantization_info);
        }
        TensorType *dst_ptr = _in_place ? &src[0] : &dst[0];
        // Create and configure function
        FunctionType norm;
        norm.configure(src[0].info(), dst_ptr->info(), epsilon);

        allocate_and_fill_tensors(src, dst);

        if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL
            std::vector<std::thread> threads;
            threads.reserve(_num_parallel_runs);

            for (int i = 0; i < _num_parallel_runs; ++i)
            {
                // Compute function
                TensorType *dst_ptr = _in_place ? &src[i] : &dst[i];
                run_pack[i]         = {{arm_compute::TensorType::ACL_SRC, &src[i]},
                                       {arm_compute::TensorType::ACL_DST, dst_ptr}};

                threads.emplace_back(
                    [&, i]
                    {
                        norm.run(run_pack[i]);
                        _target[i] = _in_place ? std::move(src[i]) : std::move(dst[i]);
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
            TensorType *dst_ptr = _in_place ? &src[0] : &dst[0];
            ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC, &src[0]},
                                 {arm_compute::TensorType::ACL_DST, dst_ptr}};
            norm.run(run_pack);

            _target[0] = _in_place ? std::move(src[0]) : std::move(dst[0]);
        }
    }

    void compute_reference(const TensorShape &shape, float epsilon)
    {
        // Create reference
        SimpleTensor<T> ref_src{shape, _data_type, 1, _input_quantization_info};
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            // Fill reference
            fill(ref_src, i);

            _reference[i] = reference::mean_std_normalization_layer<T>(ref_src, epsilon, _output_quantization_info);
        }
    }

protected:
    TensorType       _target[NUM_THREADS];
    SimpleTensor<T>  _reference[NUM_THREADS];
    bool             _in_place{};
    TestType         _test_type{};
    int              _num_parallel_runs{};
    QuantizationInfo _input_quantization_info{};
    QuantizationInfo _output_quantization_info{};
    DataType         _data_type{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuMeanStdDevNormalizationValidationFixture : public CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, bool in_place, float epsilon, DataType data_type)
    {
        CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, epsilon, data_type, QuantizationInfo(), TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuMeanStdDevNormalizationFloatThreadSafeValidationFixture : public CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, bool in_place, float epsilon, DataType data_type)
    {
        CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, epsilon, data_type, QuantizationInfo(), TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuMeanStdDevNormalizationQuantizedThreadSafeValidationFixture : public CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, bool in_place, float epsilon, DataType data_type, QuantizationInfo qinfo)
    {
        CpuMeanStdDevNormalizationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, epsilon, data_type, qinfo, TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUMEANSTDDEVNORMALIZATIONFIXTURE_H
