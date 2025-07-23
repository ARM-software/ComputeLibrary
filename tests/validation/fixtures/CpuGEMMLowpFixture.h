/*
 * Copyright (c) 2017-2025 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H

#include "tests/validation/fixtures/GEMMLowpFixture.h"

#include <cstdint>

#ifndef BARE_METAL
#include <thread>
#endif // ifndef BARE_METAL

namespace arm_compute
{
namespace test
{
namespace validation
{

namespace {
constexpr int NUM_THREADS = 3;

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d, bool reinterpret_output_as_3d, typename OutputType, bool is_fused = false, bool run_twice = false>
void compute_cpugemmlowp_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo,
                                   const QuantizationInfo& output_qinfo, DataType data_type_a = DataType::QASYMM8, DataType data_type_b = DataType::QASYMM8,
                                   GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo(), bool reshape_b_only_on_first_run = false, const TensorFillInfo& finfo = TensorFillInfo(),
                                   bool accumulate = false, bool dynamic_qinfo = false, DataType data_type_output = DataType::UNKNOWN, int num_parallel_runs = 1, TensorType targets[NUM_THREADS] = {})
{
    ARM_COMPUTE_ASSERT(is_data_type_quantized_asymmetric(data_type_a));
    ARM_COMPUTE_ASSERT(num_parallel_runs > 1 ? run_twice == false : true);

    // If unknown, set to sensible defaults
    if (data_type_output == DataType::UNKNOWN) {
        data_type_output = output_stage.type == GEMMLowpOutputStageType::NONE ? DataType::S32 : data_type_a;
    }

    // Create tensors
    TensorType a[NUM_THREADS];
    TensorType b[NUM_THREADS];
    TensorType output[NUM_THREADS];
    TensorType *out_ptrs[NUM_THREADS];
    TensorType bias[NUM_THREADS];

    for(int i = 0; i < num_parallel_runs; ++i){
        a[i] = create_tensor<TensorType>(shape_a, data_type_a, 1, dynamic_qinfo ? QuantizationInfo(1.0,0,true) : a_qinfo);
        b[i] = create_tensor<TensorType>(shape_b, data_type_b, 1, dynamic_qinfo ? QuantizationInfo(1.0,0,true) : b_qinfo); // gemm output before output stage mismatch if i pass data_layout_output here. to be investigated
        output[i] = create_tensor<TensorType>(shape_output, data_type_output, 1, output_qinfo /* output_qinfo will be ignored when output stage type is None */);
        out_ptrs[i] = &output[i];

        if(is_fused)
        {
            TensorShape bias_shape(shape_b[0]);
            bias[i] = create_tensor<TensorType>(bias_shape,data_type_output == DataType::F32 ? DataType::F32 : DataType::S32, 1);
        }
    }

    // Create and configure function
    // The GEMMinfo includes the values of the depth in case of reinterpreted 3d input/output
    FunctionType gemmlowp;
    gemmlowp.configure(a[0].info(), b[0].info(), is_fused ? bias[0].info() : nullptr, out_ptrs[0]->info(), GEMMInfo(false, false, reshape_b_only_on_first_run, (reinterpret_output_as_3d ? shape_output[2] : 0), reinterpret_input_as_3d, false,
                                                                             output_stage, false /*fp_mixed_precision*/, false /*fast_math*/, false /*broadcast_bias*/,
                                                                             arm_compute::ActivationLayerInfo(), false /* fixed_format */, arm_compute::WeightFormat::UNSPECIFIED,
                                                                             false /* pretranspose_B */, accumulate));

    for(int i = 0; i < num_parallel_runs; ++i)
    {
        // If the QuantizationInfo is dynamic, it needs to be settable after configure (note that we also force it to be dynamic)
        if (dynamic_qinfo)
        {
            a[i].info()->set_quantization_info(QuantizationInfo(a_qinfo.scale(), a_qinfo.offset(), true));
            b[i].info()->set_quantization_info(QuantizationInfo(b_qinfo.scale(), b_qinfo.offset(), true));
            output[i].info()->set_quantization_info(QuantizationInfo(output_qinfo.scale(), output_qinfo.offset(), true));
            gemmlowp.update_quantization_parameters(a[i].info()->quantization_info(),
                                                    b[i].info()->quantization_info(),
                                                    output[i].info()->quantization_info(),
                                                    data_type_output,
                                                    true, true);
        }

        ARM_COMPUTE_ASSERT(a[i].info()->is_resizable());
        ARM_COMPUTE_ASSERT(b[i].info()->is_resizable());
        ARM_COMPUTE_ASSERT(output[i].info()->is_resizable());

        add_padding_x({ &a[i], &b[i], &output[i] });

        // Allocate tensors
        a[i].allocator()->allocate();
        b[i].allocator()->allocate();
        output[i].allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a[i].info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b[i].info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output[i].info()->is_resizable());
    }

    ITensorPack pack [NUM_THREADS];

#ifndef BARE_METAL
    std::vector<std::thread> threads;

    if(num_parallel_runs > 1)
    {
        threads.reserve(num_parallel_runs);
    }
#endif // ifndef BARE_METAL

    for(int i = 0; i < num_parallel_runs; ++i)
    {
        // these are newly created every call of this lambda function
        pack[i] =
        {
            { arm_compute::TensorType::ACL_SRC_0, &a[i] },
            { arm_compute::TensorType::ACL_SRC_1, &b[i] },
            { arm_compute::TensorType::ACL_DST, out_ptrs[i] }
        };

        // Fill tensors
        fill_quantized(AccessorType(a[i]), 0 + finfo.hash);
        fill_quantized(AccessorType(b[i]), 1 + finfo.hash);

        if (accumulate)
        {
            ARM_COMPUTE_ASSERT(accumulate != run_twice);
            fill(AccessorType(output[i]), 6 + finfo.hash, finfo.min_output, finfo.max_output);
        }

        if(is_fused)
        {
            ARM_COMPUTE_ASSERT(bias[i].info()->is_resizable());
            bias[i].allocator()->allocate();
            ARM_COMPUTE_ASSERT(!bias[i].info()->is_resizable());
            fill(AccessorType(bias[i]), 2 + finfo.hash, finfo.min_bias, finfo.max_bias);
            pack[i].add_tensor(arm_compute::TensorType::ACL_SRC_2, &bias[i]);
        }

        // Run with variable inputs.
        if(run_twice)
        {
            auto mg = MemoryGroup{};
            auto ws = manage_workspace<Tensor>(gemmlowp.workspace(), mg, pack[i], pack[i]);

            gemmlowp.run(pack[i]);
            fill_quantized(AccessorType(a[i]), 3 + finfo.hash); // Fill tensors with new seed after run
            fill_quantized(AccessorType(b[i]), 4 + finfo.hash);
            if(is_fused)
            {
                fill(AccessorType(bias[i]), 5 + finfo.hash, finfo.min_bias, finfo.max_bias);
            }
        }

        // Compute GEMM function
#ifndef BARE_METAL
        if(num_parallel_runs > 1)
        {
            threads.emplace_back([&,i]
            {
                auto mg = MemoryGroup{};
                auto ws = manage_workspace<Tensor>(gemmlowp.workspace(), mg, pack[i], pack[i]);

                gemmlowp.run(pack[i]);
                targets[i] =std::move(*(out_ptrs[i]));
            });
        }
        else
#endif // ifndef BARE_METAL
        {
            auto mg = MemoryGroup{};
            auto ws = manage_workspace<Tensor>(gemmlowp.workspace(), mg, pack[i], pack[i]);

            gemmlowp.run(pack[i]);
            targets[i] = std::move(*(out_ptrs[i]));
        }
    }

#ifndef BARE_METAL
    if(num_parallel_runs > 1)
    {
        for(int i = 0; i < num_parallel_runs; ++i)
        {
            threads[i].join();
        }
    }
#endif // ifndef BARE_METAL
}
} // namespace

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, bool run_twice = false>
class CpuGEMMLowpMatrixMultiplyCoreValidationFixture : protected GEMMLowpGenericMatrixMultiplyCoreValidationFixture<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, run_twice>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset)
    {
        const auto a_qinfo = QuantizationInfo(1.0f / 255, a_offset);
        const auto b_qinfo = QuantizationInfo(2.0f / 255, b_offset);
        TensorFillInfo finfo;

        bool accumulate = false;
        bool dynamic_qinfo = false;
        this->_num_parallel_runs = 1;
        compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate, dynamic_qinfo);
        this->_references[0] = this->compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate);
    }

protected:
    void compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo, const TensorFillInfo& finfo, const bool accumulate, const bool dynamic_qinfo)
    {
        const auto output_qinfo = QuantizationInfo(); // No output stage
        compute_cpugemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, int32_t, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo, DataType::QASYMM8, DataType::QASYMM8, GEMMLowpOutputStageInfo(), false, finfo, accumulate, dynamic_qinfo, DataType::UNKNOWN, this->_num_parallel_runs, this->_targets);
    }

    int                   _num_parallel_runs{};
    TensorType            _targets[NUM_THREADS];
    SimpleTensor<int32_t> _references[NUM_THREADS];
};

template <typename TensorType, typename AccessorType, typename FunctionType, bool reinterpret_input_as_3d = false, bool reinterpret_output_as_3d = false, bool run_twice = false>
class CpuGEMMLowpStaticQuantMatrixMultiplyCoreValidationFixture : protected CpuGEMMLowpMatrixMultiplyCoreValidationFixture<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, run_twice>
{
public:
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_output, int32_t a_offset, int32_t b_offset, DataType data_type, bool is_multithreaded)
    {
        ARM_COMPUTE_ASSERT(data_type == DataType::QASYMM8_SIGNED || data_type == DataType::QASYMM8);
        const auto a_qinfo = QuantizationInfo(1.0f / 255, a_offset);
        const auto b_qinfo = QuantizationInfo(2.0f / 255, b_offset);
        TensorFillInfo finfo;

        bool accumulate = false;
        bool dynamic_qinfo = true;
        this->_num_parallel_runs = is_multithreaded ? NUM_THREADS : 1;
        compute_target(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, accumulate, dynamic_qinfo, data_type);
        compute_reference(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, finfo, data_type);
    }

protected:
    void compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo, const TensorFillInfo& finfo, const bool accumulate, const bool dynamic_qinfo, const DataType data_type)
    {
        const auto output_qinfo = QuantizationInfo(a_qinfo.scale(), a_qinfo.offset()); // No output stage
        compute_cpugemmlowp_target<TensorType, AccessorType, FunctionType, reinterpret_input_as_3d, reinterpret_output_as_3d, int32_t, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, output_qinfo, data_type, data_type, GEMMLowpOutputStageInfo(), false, finfo, accumulate, dynamic_qinfo, DataType::UNKNOWN, this->_num_parallel_runs, this->_targets);
    }

    void compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_output, const QuantizationInfo& a_qinfo, const QuantizationInfo& b_qinfo, const TensorFillInfo& finfo, const DataType data_type)
    {
        for(int i = 0; i < this->_num_parallel_runs; ++i)
        {
            if(data_type == DataType::QASYMM8)
            {
                this->_references[i] =  compute_gemmlowp_reference<reinterpret_input_as_3d, uint8_t, uint8_t, false, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type, data_type, finfo);
            }
            else
            {
                this->_references[i] =  compute_gemmlowp_reference<reinterpret_input_as_3d, int8_t, int8_t, false, false, run_twice>(shape_a, shape_b, shape_output, a_qinfo, b_qinfo, data_type, data_type, finfo);
            }
        }
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMLOWPFIXTURE_H
