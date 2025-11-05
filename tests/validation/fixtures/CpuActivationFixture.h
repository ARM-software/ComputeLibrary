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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUACTIVATIONFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUACTIVATIONFIXTURE_H

#include "arm_compute/core/TensorShape.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/helpers/ActivationHelpers.h"
#include "tests/validation/reference/ActivationLayer.h"

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
class CpuActivationValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type,
               QuantizationInfo                        quantization_info,
               TestType                                test_type)
    {
        if (data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        ActivationLayerInfo info(function, alpha_beta, alpha_beta);

        _in_place          = in_place;
        _data_type         = data_type;
        _function          = function;
        _test_type         = test_type;
        _num_parallel_runs = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        _output_quantization_info = helper::calculate_output_quantization_info(_data_type, info, quantization_info);
        _input_quantization_info  = in_place ? _output_quantization_info : quantization_info;

        compute_target(shape, info);
        compute_reference(shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int seed_offset)
    {
        if (is_data_type_float(_data_type))
        {
            float min_bound                = 0;
            float max_bound                = 0;
            std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<T>(_function, _data_type);

            if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
            {
                // Different threads should use different values for better testing, therefore we cannot use fill_static_values()
                library->fill_tensor_uniform(tensor, seed_offset, static_cast<T>(min_bound), static_cast<T>(max_bound));
            }
            else
            {
                library->fill_static_values(tensor, helper::get_boundary_values(_data_type, static_cast<T>(min_bound),
                                                                                static_cast<T>(max_bound)));
            }
        }
        else
        {
            PixelValue min{};
            PixelValue max{};
            std::tie(min, max) = get_min_max(tensor.data_type());
            library->fill_static_values(tensor, helper::get_boundary_values(_data_type, min.get<T>(), max.get<T>()));
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

    void compute_target(const TensorShape &shape, ActivationLayerInfo info)
    {
        // Create tensors
        TensorType  src[NUM_THREADS];
        TensorType  dst[NUM_THREADS];
        ITensorPack run_pack[NUM_THREADS];

        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            src[i] = create_tensor<TensorType>(shape, _data_type, 1, _input_quantization_info, DataLayout::NCHW);
            dst[i] = create_tensor<TensorType>(shape, _data_type, 1, _output_quantization_info, DataLayout::NCHW);
        }

        // Create and configure function
        FunctionType act_layer;

        if (!_in_place)
        {
            act_layer.configure(src[0].info(), dst[0].info(), info);
        }
        else
        {
            act_layer.configure(src[0].info(), nullptr, info);
        }

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
                        act_layer.run(run_pack[i]);
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
            act_layer.run(run_pack);

            _target[0] = _in_place ? std::move(src[0]) : std::move(dst[0]);
        }
    }

    void compute_reference(const TensorShape &shape, ActivationLayerInfo &info)
    {
        // Create reference
        SimpleTensor<T> src{shape, _data_type, 1, _input_quantization_info};

        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            // Fill reference
            fill(src, i);

            _reference[i] = reference::activation_layer<T>(src, info, _output_quantization_info);
        }
    }

protected:
    TensorType                              _target[NUM_THREADS];
    SimpleTensor<T>                         _reference[NUM_THREADS];
    bool                                    _in_place{};
    TestType                                _test_type{};
    int                                     _num_parallel_runs{};
    QuantizationInfo                        _input_quantization_info{};
    QuantizationInfo                        _output_quantization_info{};
    DataType                                _data_type{};
    ActivationLayerInfo::ActivationFunction _function{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuActivationValidationFixture
    : public CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type)
    {
        CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, QuantizationInfo(), TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuActivationFloatThreadSafeValidationFixture
    : public CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type)
    {
        CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, QuantizationInfo(),
            TestType::ConfigureOnceRunMultiThreaded);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuActivationQuantizedThreadSafeValidationFixture
    : public CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape                             shape,
               bool                                    in_place,
               ActivationLayerInfo::ActivationFunction function,
               float                                   alpha_beta,
               DataType                                data_type,
               QuantizationInfo                        qinfo)
    {
        CpuActivationValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, in_place, function, alpha_beta, data_type, qinfo, TestType::ConfigureOnceRunMultiThreaded);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUACTIVATIONFIXTURE_H
