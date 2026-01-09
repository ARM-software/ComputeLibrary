/*
 * Copyright (c) 2017-2026 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUFULLYCONNECTEDFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUFULLYCONNECTEDFIXTURE_H

#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/kernels/assembly/arm_common/internal/utils.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/FullyConnectedLayer.h"

#include <random>
#include <type_traits>

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
class CpuFullyConnectedValidationGenericFixture : public framework::Fixture
{
public:
    using TDecay = typename std::decay<T>::type;
    using TBias  = typename std::
        conditional<(std::is_same<TDecay, uint8_t>::value || std::is_same<TDecay, int8_t>::value), int32_t, T>::type;

public:
    void setup(TensorShape         input_shape,
               TensorShape         weights_shape,
               TensorShape         bias_shape,
               TensorShape         output_shape,
               DataType            data_type,
               QuantizationInfo    quantization_info,
               ActivationLayerInfo activation_info,
               TestType            test_type,
               bool                with_bias = true)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        ARM_COMPUTE_UNUSED(weights_shape);
        ARM_COMPUTE_UNUSED(bias_shape);

        _data_type         = data_type;
        _bias_data_type    = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
        _test_type         = test_type;
        _num_parallel_runs = (_test_type == TestType::ConfigureOnceRunMultiThreaded ? NUM_THREADS : 1);

        _input_q_info  = quantization_info;
        _weight_q_info = quantization_info;
        _dst_q_info    = quantization_info;

        _activation_info = activation_info;

        compute_target(input_shape, weights_shape, bias_shape, output_shape, with_bias);
        compute_reference(input_shape, weights_shape, bias_shape, output_shape, with_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if (_data_type == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else if (_data_type == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    inline TensorInfo prepare_weights(const TensorInfo tensor_info, const arm_compute::WeightFormat weight_format)
    {
        const DataType    data_type    = tensor_info.data_type();
        const TensorShape tensor_shape = tensor_info.tensor_shape();
        const int         ic           = tensor_shape[0];
        const int         oc           = tensor_shape[1];

        const int interleave_by = arm_compute::interleave_by(weight_format);
        const int block_by      = arm_compute::block_by(weight_format);
        const int Ip            = arm_gemm::roundup<unsigned int>(ic, block_by);
        const int Op            = arm_gemm::roundup<unsigned int>(oc, interleave_by);

        arm_compute::Strides strides_in_bytes = tensor_info.strides_in_bytes();
        strides_in_bytes.set(1, Ip * interleave_by * tensor_info.element_size());
        strides_in_bytes.set(2, Op * Ip * tensor_info.element_size());

        const size_t offset_first_element_in_bytes = tensor_info.offset_first_element_in_bytes();

        // Total size needs to include padded dimensions
        const size_t total_size_in_bytes = Op * Ip * tensor_info.element_size();

        TensorInfo new_tensor_info = tensor_info;
        new_tensor_info.set_data_layout(arm_compute::DataLayout::UNKNOWN);
        new_tensor_info.init(arm_compute::TensorShape(Ip, Op), tensor_info.num_channels(), data_type, strides_in_bytes,
                             offset_first_element_in_bytes, total_size_in_bytes);
        return new_tensor_info;
    }

    void compute_target(const TensorShape &input_shape,
                        const TensorShape &weights_shape,
                        const TensorShape &bias_shape,
                        const TensorShape &output_shape,
                        bool               with_bias)
    {
        TensorShape reshaped_weights_shape(weights_shape);

        const size_t shape_x = reshaped_weights_shape.x();
        reshaped_weights_shape.set(0, reshaped_weights_shape.y());
        reshaped_weights_shape.set(1, shape_x);

        // Create tensors
        TensorType  src[NUM_THREADS];
        TensorType  weights[NUM_THREADS];
        TensorType  reordered_weights[NUM_THREADS];
        TensorType  bias[NUM_THREADS];
        TensorType  dst[NUM_THREADS];
        TensorType  tmp_weights;
        ITensorPack run_pack[NUM_THREADS];
        ITensorPack prep_pack[NUM_THREADS];

        // Create Fully Connected layer info
        FullyConnectedLayerInfo fc_info;
        fc_info.transpose_weights    = false;
        fc_info.are_weights_reshaped = true;
        fc_info.activation_info      = _activation_info;

        NEReorderLayer            reorder;
        arm_compute::WeightFormat computed_weight_format{arm_compute::WeightFormat::ANY};
        WeightsInfo               wei_info(false, 1, 1, weights_shape[0], false, computed_weight_format);
        wei_info.set_weight_format(computed_weight_format);

        // Create tensors
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            src[i]     = create_tensor<TensorType>(input_shape, _data_type, 1, _input_q_info);
            weights[i] = create_tensor<TensorType>(reshaped_weights_shape, _data_type, 1, _weight_q_info);
            bias[i]    = with_bias ? create_tensor<TensorType>(bias_shape, _bias_data_type, 1) : nullptr;
            dst[i]     = create_tensor<TensorType>(output_shape, _data_type, 1, _dst_q_info);
            weights[i].info()->set_are_values_constant(false);
        }
        tmp_weights = create_tensor<TensorType>(weights_shape, _data_type, 1, _weight_q_info);
        tmp_weights.allocator()->allocate();

        const bool kernel_found =
            bool(FunctionType::has_opt_impl(computed_weight_format, src[0].info(), weights[0].info(),
                                            with_bias ? bias[0].info() : nullptr, dst[0].info(), fc_info, wei_info));
        ARM_COMPUTE_ASSERT(kernel_found);
        wei_info.set_weight_format(computed_weight_format);

        auto reordered_weight_info = prepare_weights(*tmp_weights.info(), computed_weight_format);
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            reordered_weights[i] = create_tensor<TensorType>(reordered_weight_info);
            reordered_weights[i].info()->set_is_resizable(true);
        }

        // Create, configure and validate function.
        FunctionType fc;
        fc.configure(src[0].info(), weights[0].info(), with_bias ? bias[0].info() : nullptr, dst[0].info(), fc_info,
                     wei_info);
        auto const aux_mem_req = fc.workspace();

        ARM_COMPUTE_ASSERT(fc.validate(src[0].info(), weights[0].info(), with_bias ? bias[0].info() : nullptr,
                                       dst[0].info(), fc_info, wei_info));

        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            ARM_COMPUTE_ASSERT(src[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(weights[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(reordered_weights[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(dst[i].info()->is_resizable());

            // Allocate tensors
            src[i].allocator()->allocate();
            weights[i].allocator()->allocate();
            reordered_weights[i].allocator()->allocate();
            dst[i].allocator()->allocate();

            ARM_COMPUTE_ASSERT(!src[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!weights[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!reordered_weights[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!dst[i].info()->is_resizable());

            // Fill tensors
            fill(AccessorType(src[i]), 0 + i * 3);
            fill(AccessorType(tmp_weights), 1 + i * 3);

            // Handle optional bias
            if (with_bias)
            {
                ARM_COMPUTE_ASSERT(bias[i].info()->is_resizable());
                bias[i].allocator()->allocate();
                ARM_COMPUTE_ASSERT(!bias[i].info()->is_resizable());
                fill(AccessorType(bias[i]), 2 + i * 3);
            }

            // Reorder weight to the expected format
            ARM_COMPUTE_ASSERT(reorder.validate(tmp_weights.info(), reordered_weights[i].info(), WeightFormat::OHWI,
                                                computed_weight_format, false));
            reorder.configure(&tmp_weights, &reordered_weights[i], WeightFormat::OHWI, computed_weight_format, false);
            reorder.run();
        }

        // Prepare function.
        prep_pack[0].add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &reordered_weights[0]);
        prep_pack[0].add_const_tensor(arm_compute::TensorType::ACL_SRC_2, &bias[0]);
        fc.prepare(prep_pack[0]);

        if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
        {
#ifndef BARE_METAL
            std::vector<std::thread> threads;
            threads.reserve(_num_parallel_runs);

            for (int i = 0; i < _num_parallel_runs; ++i)
            {
                // Compute function
                run_pack[i].add_const_tensor(arm_compute::TensorType::ACL_SRC_0, &src[i]);
                run_pack[i].add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &reordered_weights[i]);
                run_pack[i].add_const_tensor(arm_compute::TensorType::ACL_SRC_2, &bias[i]);
                run_pack[i].add_tensor(arm_compute::TensorType::ACL_DST, &dst[i]);

                threads.emplace_back(
                    [&, i]
                    {
                        auto mg = MemoryGroup{};
                        auto ws = manage_workspace<Tensor>(aux_mem_req, mg, run_pack[i], prep_pack[i]);

                        fc.run(run_pack[i]);

                        _target[i] = std::move(dst[i]);
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
            run_pack[0].add_const_tensor(arm_compute::TensorType::ACL_SRC_0, &src[0]);
            run_pack[0].add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &reordered_weights[0]);
            run_pack[0].add_const_tensor(arm_compute::TensorType::ACL_SRC_2, &bias[0]);
            run_pack[0].add_tensor(arm_compute::TensorType::ACL_DST, &dst[0]);

            auto mg = MemoryGroup{};
            auto ws = manage_workspace<Tensor>(aux_mem_req, mg, run_pack[0], prep_pack[0]);

            fc.run(run_pack[0]);

            _target[0] = std::move(dst[0]);
        }
    }

    void compute_reference(const TensorShape &input_shape,
                           const TensorShape &weights_shape,
                           const TensorShape &bias_shape,
                           const TensorShape &output_shape,
                           bool               with_bias)
    {
        // Create reference
        SimpleTensor<T>     ref_src{input_shape, _data_type, 1, _input_q_info};
        SimpleTensor<T>     ref_weights{weights_shape, _data_type, 1, _weight_q_info};
        SimpleTensor<TBias> ref_bias{bias_shape, _bias_data_type, 1, QuantizationInfo()};

        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            // Fill reference
            fill(ref_src, 0 + i * 3);
            fill(ref_weights, 1 + i * 3);

            if (with_bias)
            {
                fill(ref_bias, 2 + i * 3);
            }

            _reference[i] =
                reference::activation_layer(reference::fully_connected_layer<T>(ref_src, ref_weights, ref_bias,
                                                                                output_shape, _dst_q_info, with_bias),
                                            _activation_info, _dst_q_info);
        }
    }

    TensorType          _target[NUM_THREADS];
    SimpleTensor<T>     _reference[NUM_THREADS];
    DataType            _data_type{};
    DataType            _bias_data_type{};
    TestType            _test_type{};
    QuantizationInfo    _input_q_info{};
    QuantizationInfo    _weight_q_info{};
    QuantizationInfo    _dst_q_info{};
    ActivationLayerInfo _activation_info{};
    int                 _num_parallel_runs{};

    int _hash{0};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuFullyConnectedValidationFixture
    : public CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape         input_shape,
               TensorShape         weights_shape,
               TensorShape         bias_shape,
               TensorShape         output_shape,
               DataType            data_type,
               ActivationLayerInfo activation_info)
    {
        CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            input_shape, weights_shape, bias_shape, output_shape, data_type, QuantizationInfo(), activation_info,
            TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuFullyConnectedValidationFixtureNoBias
    : public CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape         input_shape,
               TensorShape         weights_shape,
               TensorShape         bias_shape,
               TensorShape         output_shape,
               DataType            data_type,
               ActivationLayerInfo activation_info)
    {
        CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            input_shape, weights_shape, bias_shape, output_shape, data_type, QuantizationInfo(), activation_info,
            TestType::ConfigureOnceRunOnce, false);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuFullyConnectedThreadSafeValidationFixture
    : public CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape         input_shape,
               TensorShape         weights_shape,
               TensorShape         bias_shape,
               TensorShape         output_shape,
               DataType            data_type,
               ActivationLayerInfo activation_info)
    {
        CpuFullyConnectedValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            input_shape, weights_shape, bias_shape, output_shape, data_type, QuantizationInfo(), activation_info,
            TestType::ConfigureOnceRunMultiThreaded);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUFULLYCONNECTEDFIXTURE_H
